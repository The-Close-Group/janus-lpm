import os
import json
import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Any
from fastmcp import FastMCP, Context
from sklearn.preprocessing import StandardScaler

# Initialize a subserver for feature selection
app = FastMCP(name="FeatureSelectionService")

# Base path to locate data.json
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_JSON = os.path.join(BASE_DIR, 'data', 'data.json')

@app.tool()
def load_data() -> str:
    """
    Load raw census records and metadata from data.json into Context.storage:
      - raw_df: full DataFrame of ACS rows
      - var_meta: dict of each variable's metadata (including optional 'formula')
    """
    with open(DATA_JSON, 'r') as f:
        payload = json.load(f)

    raw = payload.get('data')
    if raw is None:
        raise RuntimeError("No 'data' key found in data.json")

    df = pd.DataFrame(raw)
    Context.storage['raw_df'] = df
    Context.storage['var_meta'] = payload.get('variables', {})
    return f"Loaded data: {df.shape[0]} rows, {len(Context.storage['var_meta'])} variables."


def _build_registry() -> Dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """
    Build feature→function map from metadata; support 'formula' or raw column.
    """
    meta = Context.storage.get('var_meta')
    if meta is None:
        raise RuntimeError("Metadata not loaded—call load_data() first.")

    registry: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
    for code, info in meta.items():
        if info.get('predicateOnly'):
            continue
        expr = info.get('formula', '').strip()
        if expr:
            registry[code] = lambda df, expr=expr: df.eval(expr)
        else:
            registry[code] = lambda df, col=code: df[col]
    return registry

@app.tool()
def list_features() -> List[str]:
    """Return sorted list of available feature codes."""
    return sorted(_build_registry().keys())

@app.tool()
def select_features(features: List[str]) -> Dict[str, List[float]]:
    """
    Validate and compute each feature in `features`, returning code→values list.
    """
    df = Context.storage.get('raw_df')
    if df is None:
        raise RuntimeError("Raw data not loaded—call load_data() first.")

    registry = _build_registry()
    invalid = [f for f in features if f not in registry]
    if invalid:
        raise ValueError(f"Unknown feature(s): {invalid}. Use list_features().")

    output: Dict[str, List[float]] = {}
    for code in features:
        output[code] = registry[code](df).astype(float).tolist()
    return output

@app.tool()
def generate_synthetic_targets(features: List[str], sample_size: int = 1000) -> Dict[str, Any]:
    """
    Generate synthetic target variables based on selected features:
      - CTR: synthetic click-through rates (0-1)
      - CPC: synthetic cost-per-click ($)
      - CONV: synthetic conversion rates (0-1)
    
    Returns X feature matrix and Y target values.
    """
    # First ensure features are selected and computed
    feature_values = select_features(features)
    
    # Create feature matrix as 2D list
    num_samples = len(list(feature_values.values())[0]) if feature_values else 0
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
    
    # Store in context for model training
    synthetic_data = {
        'X': X,
        'ctr': ctr.tolist(),
        'cpc': cpc.tolist(),
        'conv': conv.tolist(),
        'feature_names': features
    }
    Context.storage['synthetic_data'] = synthetic_data
    
    return synthetic_data

# Optionally, if this is a mounted subserver, main.py will mount it.
# To test standalone, uncomment below:
if __name__ == '__main__':
    app.run(transport='sse', host='0.0.0.0', port=8001)