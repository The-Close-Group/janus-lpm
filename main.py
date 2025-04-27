#!/usr/bin/env python3
"""
MarketingSimulator FastMCP Server

A fully dynamic Facebook/Instagram lead-generation campaign simulator built
using FastMCP. Exposes:
  - AgentCard discovery via `resource://agentcard`
  - Resources: available features & metrics
  - Tools: load_data, simulate, simulate_stream, cancel_task
Clients can call tools to configure features, weights, budget, variants, and metrics,
then receive results and actionable insights.
"""
import json
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from fastmcp import FastMCP, Context
from fastmcp.resources import TextResource

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CSV_PATH = Path(__file__).parent / 'data' / 'zcta_demographics.csv'
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Demographics file not found: {CSV_PATH}")

FEATURE_FUNCTIONS = {
    'population':    lambda df: df['B01003_001E'],
    'median_age':    lambda df: df['B01002_001E'],
    'pct_under18':   lambda df: (df['B01001_003E'] + df['B01001_027E']) / df['B01003_001E'],
    'pct18_34':      lambda df: df[[f'B01001_{i:03d}E' for i in range(5,13)] + [f'B01001_{i:03d}E' for i in range(29,37)]].sum(axis=1) / df['B01003_001E'],
    'pct35_64':      lambda df: df[[f'B01001_{i:03d}E' for i in range(13,21)] + [f'B01001_{i:03d}E' for i in range(37,45)]].sum(axis=1) / df['B01003_001E'],
    'pct65_plus':    lambda df: df[[f'B01001_{i:03d}E' for i in range(21,26)] + [f'B01001_{i:03d}E' for i in range(45,50)]].sum(axis=1) / df['B01003_001E'],
    'pct_male':      lambda df: df['B01001_002E'] / df['B01003_001E'],
    'pct_female':    lambda df: df['B01001_026E'] / df['B01003_001E'],
    'pct_white':     lambda df: df['B02001_002E'] / df['B02001_001E'],
    'pct_black':     lambda df: df['B02001_003E'] / df['B02001_001E'],
    'pct_asian':     lambda df: df['B02001_005E'] / df['B02001_001E'],
    'pct_hispanic':  lambda df: df['B03003_003E'] / df['B03003_001E'],
    'median_income': lambda df: df['B19013_001E'],
    'disp_income':   lambda df: df['B19013_001E'] / df['B25010_001E'],
    'pct_broadband': lambda df: (df['B28002_004E'] + df['B28002_007E']) / df['B28002_001E'],
}
DEFAULT_FEATURES = ['pct18_34', 'median_income', 'pct_broadband', 'disp_income']
METRICS = ['ctr', 'cpc', 'conv_rate', 'impressions', 'leads']

# AgentCard metadata published as a resource
AGENT_CARD = {
    'name': "MarketingSimulator",
    'description': "Dynamic FB/IG lead-gen campaign simulator using US Census data",
    'version': "1.0.0",
    'input_modes': ["text"],
    'output_modes': ["text","data"],
    'capabilities': {"streaming":True, "pushNotifications":False}
}

# -----------------------------------------------------------------------------
# FastMCP Server Setup
# -----------------------------------------------------------------------------
mcp = FastMCP(
    name="MarketingSimulator",
    dependencies=["pandas","torch"]
)

# Expose AgentCard as a static JSON resource
mcp.add_resource(
    TextResource(
        uri="resource://agentcard",
        name="AgentCard",
        text=json.dumps(AGENT_CARD),
        mime_type="application/json"
    )
)

# -----------------------------------------------------------------------------
# Lead-Gen Model Definition
# -----------------------------------------------------------------------------
class LeadGenModel(nn.Module):
    def __init__(self, weights: dict, intercept: float, base_cpc: float, conv_factor: float, idx: dict):
        super().__init__()
        self.weights = weights
        self.intercept = intercept
        self.base_cpc = base_cpc
        self.conv_factor = conv_factor
        self.idx = idx

    def forward(self, X: torch.Tensor):
        raw = torch.full((X.size(0),), self.intercept)
        for feat, w in self.weights.items():
            raw = raw + w * X[:, self.idx[feat]]
        ctr = torch.sigmoid(raw)
        disp = X[:, self.idx['disp_income']]
        cpc = self.base_cpc / (1.0 + disp)
        conv = ctr * self.conv_factor
        return ctr, cpc, conv

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
import pandas as pd

def get_top_zip(df: pd.DataFrame, arr: torch.Tensor):
    idx = int(arr.argmax().item())
    row = df.iloc[idx]
    return {
        'zip': row['zip code tabulation area'],
        'med_age': int(row['B01002_001E']),
        'med_inc': int(row['B19013_001E'])
    }

# -----------------------------------------------------------------------------
# Resources
# -----------------------------------------------------------------------------
@mcp.resource("features://list")
def list_features() -> list[str]:
    return list(FEATURE_FUNCTIONS.keys())

@mcp.resource("metrics://list")
def list_metrics() -> list[str]:
    return METRICS

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
@mcp.tool()
def load_data(
    features: list[str],
    ctx: Context
) -> dict:
    """
    Reads CSV and computes selected features.
    Returns a dict with arrays and metadata.
    """
    df = pd.read_csv(CSV_PATH)
    for f in features:
        df[f] = FEATURE_FUNCTIONS[f](df)
    return {
        'features': df[features].values.tolist(),
        'zip_codes': df['zip code tabulation area'].astype(str).tolist(),
        'median_age': df['B01002_001E'].tolist(),
        'median_income': df['B19013_001E'].tolist()
    }

@mcp.tool(name="simulate")
def simulate(
    data: dict,
    weights: dict,
    intercept: float,
    base_cpc: float,
    conv_factor: float,
    variants: int,
    budget: float,
    metrics: list[str]
) -> dict:
    """
    Runs lead-gen simulation synchronously.
    """
    X = torch.tensor(data['features'], dtype=torch.float32)
    idx_map = {feat:i for i,feat in enumerate(weights['feature_weights'].keys())}
    model = LeadGenModel(weights['feature_weights'], intercept, base_cpc, conv_factor, idx_map)
    ctr, cpc, conv = model(X)
    slice_budget = budget / variants
    impr = slice_budget / cpc
    leads = impr * conv
    df_meta = pd.DataFrame({
        'zip': data['zip_codes'],
        'median_age': data['median_age'],
        'median_income': data['median_income']
    })
    results = {}
    for i in range(variants):
        key = f"Variant_{i+1}"
        stats = {
            'ctr': ctr.mean().item(),
            'cpc': cpc.mean().item(),
            'conv_rate': conv.mean().item(),
            'impressions': int(impr.sum().item()),
            'leads': int(leads.sum().item())
        }
        for m in metrics:
            arr = {'ctr':ctr, 'cpc':cpc, 'conv_rate':conv, 'impressions':impr, 'leads':leads}[m]
            top = get_top_zip(df_meta, arr)
            stats.update({f'top_{m}_{k}':v for k,v in top.items()})
        results[key] = stats
    return results

@mcp.tool(name="simulate_stream")
def simulate_stream(
    data: dict,
    weights: dict,
    intercept: float,
    base_cpc: float,
    conv_factor: float,
    variants: int,
    budget: float,
    metrics: list[str]
):
    # progress
    total = variants
    for i in range(1, total+1):
        yield {'status': {'state': 'working', 'progress': i/total}}
    # final
    final_res = simulate(data, weights, intercept, base_cpc, conv_factor, variants, budget, metrics)
    yield {'artifact': {'name': 'results', 'parts': [], 'append': False, 'lastChunk': True, 'body': final_res}}

@mcp.tool()
def cancel_task(task_id: str) -> dict:
    return {'status': 'canceled'}

# -----------------------------------------------------------------------------
# Run Server
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    mcp.run(host="0.0.0.0", port=8000)