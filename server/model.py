import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Any, Dict, List
from fastmcp import FastMCP, Context
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Lead Generation Neural Network Model Definition
class LeadGenNet(nn.Module):
    """
    Neural network for lead generation prediction with three outputs:
    - CTR (Click-Through Rate)
    - CPC (Cost Per Click)
    - Conv (Conversion Rate)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Three output heads for CTR, CPC, and Conversion
        self.ctr_head = nn.Linear(hidden_dim, 1)
        self.cpc_head = nn.Linear(hidden_dim, 1)
        self.conv_head = nn.Linear(hidden_dim, 1)
        
        # Activation functions for each output
        self.ctr_act = nn.Sigmoid()  # CTR between 0 and 1
        self.conv_act = nn.Sigmoid()  # Conversion rate between 0 and 1
        
    def forward(self, x):
        features = self.backbone(x)
        ctr = self.ctr_act(self.ctr_head(features))
        cpc = self.cpc_head(features)  # No activation, direct prediction
        conv = self.conv_act(self.conv_head(features))
        
        # Return as batch x 3 tensor [ctr, cpc, conv]
        return torch.cat([ctr, cpc, conv], dim=1)

# Predefined model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'small':  {'hidden_dim': 32},
    'medium': {'hidden_dim': 64},
    'large':  {'hidden_dim': 128},
    'huge':   {'hidden_dim': 256},
}

# FastMCP subserver for multi-size model training, evaluation, and backtesting
app = FastMCP(name="ModelService")

@app.tool()
def list_models() -> List[str]:
    """List available model size keys."""
    return list(MODEL_CONFIGS.keys())

@app.tool()
def build_model(size: str, input_dim: int, dropout: float = 0.1) -> Dict[str, Any]:
    """
    Instantiate the specified model and return its parameter count.
    """
    cfg = MODEL_CONFIGS.get(size)
    if cfg is None:
        raise ValueError(f"Unknown model size '{size}'. Available: {list(MODEL_CONFIGS.keys())}")

    model = LeadGenNet(
        input_dim=input_dim,
        hidden_dim=cfg['hidden_dim'],
        dropout=dropout
    )
    num_params = sum(p.numel() for p in model.parameters())
    return {'size': size, 'num_parameters': num_params}

@app.tool()
def train_model(
    size: str,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    lambda_ctr: float = 0.6,
    lambda_conv: float = 0.4,
    patience: int = 5
) -> Dict[str, Any]:
    """
    Full training pipeline with evaluation metrics:
      - 80/20 train/val split
      - AdamW optimizer
      - CosineAnnealingLR scheduler
      - Weighted MSE loss on CTR & Conv + MAE on CPC
      - Early stopping & checkpointing
      - Evaluation metrics: RMSE(CTR), RMSE(Conv), MAE(CPC), R2(CTR)

    Returns best epoch, metrics, and checkpoint path.
    """
    # Fetch synthetic data
    data = Context.storage.get('synthetic_data')
    if data is None:
        raise RuntimeError("No synthetic_data found. Run generate_synthetic_targets() first.")

    X = data['X']
    y_ctr = data['ctr']
    y_cpc = data['cpc']
    y_conv = data['conv']
    
    # Convert to numpy arrays
    import numpy as np
    X = np.array(X, dtype=np.float32)
    y_ctr = np.array(y_ctr, dtype=np.float32)
    y_cpc = np.array(y_cpc, dtype=np.float32)
    y_conv = np.array(y_conv, dtype=np.float32)
    y = np.stack([y_ctr, y_cpc, y_conv], axis=1)

    # Dataset and split
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    val_count = int(len(ds) * 0.2)
    train_count = len(ds) - val_count
    train_ds, val_ds = random_split(ds, [train_count, val_count])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model instantiation
    cfg = MODEL_CONFIGS.get(size)
    
    model = LeadGenNet(input_dim=X.shape[1], hidden_dim=cfg['hidden_dim'])

    # Optimizer, scheduler, loss definitions
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    best_val_loss = float('inf')
    best_epoch = 0
    ckpt_path = f"model_{size}_best.pt"

    # Training loop with early stopping
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            preds = model(xb)
            # Weighted MSE for CTR & Conv
            loss_mse = lambda_ctr * mse_loss(preds[:, 0], yb[:, 0]) + \
                       lambda_conv * mse_loss(preds[:, 2], yb[:, 2])
            # MAE for CPC
            loss_mae = mae_loss(preds[:, 1], yb[:, 1])
            loss = loss_mse + 0.1 * loss_mae

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation to update best_val_loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss_mse = lambda_ctr * mse_loss(preds[:, 0], yb[:, 0]) + \
                           lambda_conv * mse_loss(preds[:, 2], yb[:, 2])
                loss_mae = mae_loss(preds[:, 1], yb[:, 1])
                val_losses.append((loss_mse + 0.1 * loss_mae).item() * xb.size(0))
        val_loss = sum(val_losses) / len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
        elif epoch - best_epoch >= patience:
            break

    # Load best model and evaluate
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    # Use full synthetic dataset for backtest
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch_size)
    preds_list, true_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds_list.append(model(xb).cpu().numpy())
            true_list.append(yb.cpu().numpy())
    preds = np.vstack(preds_list)
    trues = np.vstack(true_list)

    # Compute final metrics
    rmse_ctr = mean_squared_error(trues[:,0], preds[:,0], squared=False)
    rmse_conv = mean_squared_error(trues[:,2], preds[:,2], squared=False)
    mae_cpc = mean_absolute_error(trues[:,1], preds[:,1])
    r2_ctr = r2_score(trues[:,0], preds[:,0])

    return {
        'best_epoch': best_epoch,
        'rmse_ctr': rmse_ctr,
        'rmse_conv': rmse_conv,
        'mae_cpc': mae_cpc,
        'r2_ctr': r2_ctr,
        'checkpoint_path': ckpt_path
    }

@app.tool()
def backtest_model(
    size: str,
    batch_size: int = 128
) -> Dict[str, float]:
    """
    Back-test the best model checkpoint on the full synthetic dataset.
    Returns metrics: RMSE(CTR), RMSE(Conv), MAE(CPC), R2(CTR).
    """
    data = Context.storage.get('synthetic_data')
    if data is None:
        raise RuntimeError("No synthetic_data found. Run generate_synthetic_targets() first.")

    # Convert to numpy arrays
    import numpy as np
    X = np.array(data['X'], dtype=np.float32)
    y_ctr = np.array(data['ctr'], dtype=np.float32)
    y_cpc = np.array(data['cpc'], dtype=np.float32)
    y_conv = np.array(data['conv'], dtype=np.float32)
    y = np.stack([y_ctr, y_cpc, y_conv], axis=1)

    cfg = MODEL_CONFIGS.get(size)
    if cfg is None:
        raise ValueError(f"Unknown model size '{size}'")
    model = LeadGenNet(input_dim=X.shape[1], hidden_dim=cfg['hidden_dim'])
    ckpt_path = f"model_{size}_best.pt"
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch_size)
    preds_list, true_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds_list.append(model(xb).cpu().numpy())
            true_list.append(yb.cpu().numpy())
    preds = np.vstack(preds_list)
    trues = np.vstack(true_list)

    return {
        'rmse_ctr': mean_squared_error(trues[:,0], preds[:,0], squared=False),
        'rmse_conv': mean_squared_error(trues[:,2], preds[:,2], squared=False),
        'mae_cpc': mean_absolute_error(trues[:,1], preds[:,1]),
        'r2_ctr': r2_score(trues[:,0], preds[:,0])
    }

# To serve standalone:
if __name__ == '__main__':
    app.run(transport='sse', host='0.0.0.0', port=8002)