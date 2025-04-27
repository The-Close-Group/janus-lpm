# server/feature_selection.py
import os
import json
import pandas as pd
from typing import Callable, Dict, List
from fastmcp import FastMCP, Context

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

# Optionally, if this is a mounted subserver, main.py will mount it.
# To test standalone, uncomment below:
if __name__ == '__main__':
    app.run(transport='sse', host='0.0.0.0', port=8001)
