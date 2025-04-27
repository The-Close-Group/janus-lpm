#!/usr/bin/env python3
"""
Standalone script to train a lead generation model and perform inference
without FastMCP dependencies.
"""
import json
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Define the LeadGenNet model directly
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

# Load data
def load_data():
    """Load and prepare data for model training."""
    print("Loading data...")
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'data.json')
    
    with open(data_path, 'r') as f:
        payload = json.load(f)
    
    raw = payload.get('data')
    if raw is None:
        raise RuntimeError("No 'data' key found in data.json")
    
    df = pd.DataFrame(raw)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    return df, payload.get('variables', {})

# Generate synthetic targets based on selected features
def generate_synthetic_targets(df, features, sample_size=1000):
    """Generate synthetic targets for training."""
    print(f"Generating synthetic targets based on {len(features)} features...")
    
    # Validate features
    valid_features = [f for f in features if f in df.columns]
    if len(valid_features) != len(features):
        missing = set(features) - set(valid_features)
        print(f"Warning: {len(missing)} features not found: {missing}")
        features = valid_features
    
    # Extract features from dataframe
    feature_values = {f: df[f].fillna(0).astype(float).tolist() for f in features}
    
    # Create feature matrix
    num_samples = len(df)
    if sample_size > 0 and sample_size < num_samples:
        indices = np.random.choice(num_samples, size=sample_size, replace=False)
        X = [[feature_values[feat][i] for feat in features] for i in indices]
    else:
        X = [[feature_values[feat][i] for feat in features] for i in range(num_samples)]
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic targets with some relationship to features
    weights = np.random.uniform(-1, 1, size=len(features))
    intercept_ctr = np.random.uniform(0.01, 0.1)  # Base CTR
    intercept_cpc = np.random.uniform(0.5, 2.0)   # Base CPC
    intercept_conv = np.random.uniform(0.001, 0.05)  # Base conversion rate
    
    # Apply random weights
    X_arr = np.array(X)
    weighted_sum = X_arr.dot(weights)
    
    # Scale to appropriate ranges with logistic function for rates
    ctr = 1 / (1 + np.exp(-weighted_sum * 0.5 + intercept_ctr))
    cpc = np.abs(weighted_sum * 0.5 + intercept_cpc)  # Keep positive
    conv = 1 / (1 + np.exp(-weighted_sum * 0.3 + intercept_conv))
    
    # Add some noise
    ctr += np.random.normal(0, 0.02, size=len(ctr))
    cpc += np.random.normal(0, 0.2, size=len(cpc))
    conv += np.random.normal(0, 0.005, size=len(conv))
    
    # Clip to valid ranges
    ctr = np.clip(ctr, 0.001, 0.99)
    cpc = np.clip(cpc, 0.1, 10.0)
    conv = np.clip(conv, 0.0001, 0.3)
    
    synthetic_data = {
        'X': X,
        'ctr': ctr.tolist(),
        'cpc': cpc.tolist(),
        'conv': conv.tolist(),
        'feature_names': features
    }
    
    print(f"Generated synthetic data with {len(X)} samples")
    return synthetic_data

# Train the model
def train_model(synth_data, size="medium", epochs=10, batch_size=128):
    """Train the model and return the trained model and metrics."""
    print(f"Training {size} model for {epochs} epochs...")
    
    # Model configurations
    MODEL_CONFIGS = {
        'small':  {'hidden_dim': 32},
        'medium': {'hidden_dim': 64},
        'large':  {'hidden_dim': 128},
        'huge':   {'hidden_dim': 256},
    }
    
    cfg = MODEL_CONFIGS.get(size)
    if cfg is None:
        raise ValueError(f"Unknown model size: {size}")
    
    # Prepare data
    X = np.array(synth_data['X'], dtype=np.float32)
    y_ctr = np.array(synth_data['ctr'], dtype=np.float32)
    y_cpc = np.array(synth_data['cpc'], dtype=np.float32)
    y_conv = np.array(synth_data['conv'], dtype=np.float32)
    y = np.stack([y_ctr, y_cpc, y_conv], axis=1)
    
    # Create PyTorch datasets
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    val_count = int(len(ds) * 0.2)
    train_count = len(ds) - val_count
    train_ds, val_ds = random_split(ds, [train_count, val_count])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Initialize model
    model = LeadGenNet(input_dim=X.shape[1], hidden_dim=cfg['hidden_dim'])
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    
    # Training state
    best_val_loss = float('inf')
    best_epoch = 0
    ckpt_path = f"model_{size}_best.pt"
    
    # Training loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_losses = []
        
        for xb, yb in train_loader:
            preds = model(xb)
            loss_mse = 0.6 * mse_loss(preds[:, 0], yb[:, 0]) + 0.4 * mse_loss(preds[:, 2], yb[:, 2])
            loss_mae = mae_loss(preds[:, 1], yb[:, 1])
            loss = loss_mse + 0.1 * loss_mae
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss_mse = 0.6 * mse_loss(preds[:, 0], yb[:, 0]) + 0.4 * mse_loss(preds[:, 2], yb[:, 2])
                loss_mae = mae_loss(preds[:, 1], yb[:, 1])
                val_losses.append((loss_mse + 0.1 * loss_mae).item())
                
        # Calculate average losses
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model at epoch {epoch} with val loss: {val_loss:.4f}")
        
        scheduler.step()
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds. Best epoch: {best_epoch}/{epochs}")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    
    # Evaluate on full dataset
    full_loader = DataLoader(ds, batch_size=batch_size)
    all_preds, all_true = [], []
    
    with torch.no_grad():
        for xb, yb in full_loader:
            preds = model(xb)
            all_preds.append(preds.numpy())
            all_true.append(yb.numpy())
    
    # Stack predictions and true values
    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)
    
    # Calculate metrics
    metrics = {
        'best_epoch': best_epoch,
        'rmse_ctr': float(mean_squared_error(all_true[:, 0], all_preds[:, 0], squared=False)),
        'rmse_conv': float(mean_squared_error(all_true[:, 2], all_preds[:, 2], squared=False)),
        'mae_cpc': float(mean_absolute_error(all_true[:, 1], all_preds[:, 1])),
        'r2_ctr': float(r2_score(all_true[:, 0], all_preds[:, 0])),
        'checkpoint_path': ckpt_path,
        'training_time': training_time
    }
    
    print("\nTraining metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return model, metrics, synth_data

# Inference function
def perform_inference(model, synth_data, num_samples=5):
    """Perform inference on a few samples and print results."""
    print(f"\nPerforming inference on {num_samples} samples...")
    
    # Get feature names
    feature_names = synth_data['feature_names']
    
    # Prepare some test samples
    X = np.array(synth_data['X'], dtype=np.float32)
    y_ctr = np.array(synth_data['ctr'], dtype=np.float32)
    y_cpc = np.array(synth_data['cpc'], dtype=np.float32)
    y_conv = np.array(synth_data['conv'], dtype=np.float32)
    
    # Take a few samples for demonstration
    indices = np.random.choice(len(X), num_samples, replace=False)
    X_samples = X[indices]
    y_ctr_samples = y_ctr[indices]
    y_cpc_samples = y_cpc[indices]
    y_conv_samples = y_conv[indices]
    
    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X_samples)).numpy()
    
    # Print comparison
    print("\nInference Results:")
    print("-" * 80)
    print(f"{'Feature':>15} | {'True CTR':>10} | {'Pred CTR':>10} | {'True CPC':>10} | {'Pred CPC':>10} | {'True Conv':>10} | {'Pred Conv':>10}")
    print("-" * 80)
    
    for i in range(num_samples):
        print(f"Sample {i+1:>8} | {y_ctr_samples[i]:.4f} | {predictions[i, 0]:.4f} | "
              f"{y_cpc_samples[i]:.4f} | {predictions[i, 1]:.4f} | "
              f"{y_conv_samples[i]:.4f} | {predictions[i, 2]:.4f}")
    
    print("-" * 80)
    
    # Calculate and print error metrics
    ctr_error = np.mean(np.abs(y_ctr_samples - predictions[:, 0])) * 100
    cpc_error = np.mean(np.abs(y_cpc_samples - predictions[:, 1]))
    conv_error = np.mean(np.abs(y_conv_samples - predictions[:, 2])) * 100
    
    print(f"\nAverage Error Metrics:")
    print(f"CTR Error: {ctr_error:.2f}%")
    print(f"CPC Error: ${cpc_error:.2f}")
    print(f"Conversion Error: {conv_error:.2f}%")
    
    # Simulate a marketing campaign
    budget = 10000.0
    cpc_avg = np.mean(predictions[:, 1])
    ctr_avg = np.mean(predictions[:, 0])
    conv_avg = np.mean(predictions[:, 2])
    
    impressions = int(budget / cpc_avg * 1000)
    clicks = int(impressions * ctr_avg)
    conversions = int(clicks * conv_avg)
    
    print(f"\nSimulated Campaign Results (Budget: ${budget:.2f}):")
    print(f"Estimated Impressions: {impressions:,}")
    print(f"Estimated Clicks: {clicks:,}")
    print(f"Estimated Conversions: {conversions:,}")
    print(f"Cost Per Conversion: ${budget/conversions:.2f}")
    print(f"ROI (assuming $100 per conversion): {conversions*100/budget:.2f}x")

# Main function
def main():
    # 1. Load the data
    df, variables = load_data()
    
    # 2. Select the first 5 numeric features
    numeric_features = []
    for col in df.columns:
        try:
            df[col].astype(float)
            numeric_features.append(col)
            if len(numeric_features) >= 5:
                break
        except:
            continue
    
    if not numeric_features:
        # If no numeric features found, use first 5 columns
        numeric_features = df.columns[:5].tolist()
    
    print(f"Selected features: {numeric_features}")
    
    # 3. Generate synthetic targets
    synth_data = generate_synthetic_targets(df, numeric_features, sample_size=2000)
    
    # 4. Train the model
    model, metrics, synth_data = train_model(synth_data, size="medium", epochs=10, batch_size=128)
    
    # 5. Perform inference
    perform_inference(model, synth_data, num_samples=5)

if __name__ == "__main__":
    main()