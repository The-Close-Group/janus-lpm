#!/usr/bin/env python3
"""
MarketingSimulator: an MCP agent for fully dynamic FB/IG lead-gen campaign simulation
using US Census ZCTA data with customizable features, weights, metrics, and actionable insights.
"""
import os
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from mcp import Tool, Agent, run_agent
from agent_torch.data.census.census_loader import CensusDataLoader
from agent_torch.core.dataloader import LoadPopulation

# -----------------------------------------------------------------------------
# Feature definitions: map user-friendly names to DataFrame lambdas
# -----------------------------------------------------------------------------
FEATURE_FUNCTIONS = {
    'population':       lambda df: df['B01003_001E'],
    'median_age':       lambda df: df['B01002_001E'],
    'pct_under18':      lambda df: (df['B01001_003E'] + df['B01001_027E']) / df['B01003_001E'],
    'pct18_34':         lambda df: df[[f'B01001_{i:03d}E' for i in range(5,13)] + [f'B01001_{i:03d}E' for i in range(29,37)]].sum(axis=1) / df['B01003_001E'],
    'pct35_64':         lambda df: df[[f'B01001_{i:03d}E' for i in range(13,21)] + [f'B01001_{i:03d}E' for i in range(37,45)]].sum(axis=1) / df['B01003_001E'],
    'pct65_plus':       lambda df: df[[f'B01001_{i:03d}E' for i in range(21,26)] + [f'B01001_{i:03d}E' for i in range(45,50)]].sum(axis=1) / df['B01003_001E'],
    'pct_male':         lambda df: df['B01001_002E'] / df['B01003_001E'],
    'pct_female':       lambda df: df['B01001_026E'] / df['B01003_001E'],
    'pct_white':        lambda df: df['B02001_002E'] / df['B02001_001E'],
    'pct_black':        lambda df: df['B02001_003E'] / df['B02001_001E'],
    'pct_asian':        lambda df: df['B02001_005E'] / df['B02001_001E'],
    'pct_hispanic':     lambda df: df['B03003_003E'] / df['B03003_001E'],
    'median_income':    lambda df: df['B19013_001E'],
    'disp_income':      lambda df: df['B19013_001E'] / df['B25010_001E'],
    'pct_broadband':    lambda df: (df['B28002_004E'] + df['B28002_007E']) / df['B28002_001E'],
    # add more as needed...
}
DEFAULT_FEATURES = ['pct18_34','median_income','pct_broadband','disp_income']
METRICS = ['ctr','cpc','conv_rate','impressions','leads']
CSV_PATH = Path(__file__).parent / 'data' / 'zcta_demographics.csv'
if not CSV_PATH.exists(): raise FileNotFoundError(f"Missing {CSV_PATH}")

# -----------------------------------------------------------------------------
# Dynamic Lead-Generation Model
# -----------------------------------------------------------------------------
class LeadGenModel(nn.Module):
    """
    Customizable linear model for CTR; CPC and conv_rate formulas fixed but can be extended.
    """
    def __init__(self, feature_weights: dict, intercept: float, base_cpc: float, conv_factor: float, feature_idx: dict):
        super().__init__()
        self.feature_weights = feature_weights
        self.intercept = intercept
        self.base_cpc = base_cpc
        self.conv_factor = conv_factor
        self.feature_idx = feature_idx

    def forward(self, X: torch.Tensor):
        # CTR: intercept + sum(w_i * X[:, idx_i]) passed through sigmoid
        raw = self.intercept
        for feat, w in self.feature_weights.items():
            idx = self.feature_idx[feat]
            raw = raw + w * X[:, idx]
        ctr = torch.sigmoid(raw)
        # CPC inversely ~ disp_income
        disp = X[:, self.feature_idx['disp_income']]
        cpc = self.base_cpc / (1.0 + disp)
        # conversion rate
        conv = ctr * self.conv_factor
        return ctr, cpc, conv

# -----------------------------------------------------------------------------
# Helper: identify top ZIP for a metric
# -----------------------------------------------------------------------------

def get_top_zip(df: pd.DataFrame, arr: torch.Tensor):
    vals = arr.detach().cpu().numpy()
    idx = int(vals.argmax())
    row = df.iloc[idx]
    return row['zip code tabulation area'], row['B01002_001E'], row['B19013_001E']

# -----------------------------------------------------------------------------
# MCP Agent
# -----------------------------------------------------------------------------
class MarketingSimulator(Agent):
    @Tool
    def ask_business(self): return self.prompt("Describe your B2B SMB offering:")
    @Tool
    def ask_budget(self): return float(self.prompt("Monthly ad budget (USD)?"))
    @Tool
    def ask_variants(self): k=int(self.prompt("How many campaign variants to test? (3-5)")); return max(3,min(k,5))

    @Tool
    def ask_fields(self):
        text = self.prompt(
            f"Select features to include (comma-separated). Options: {', '.join(FEATURE_FUNCTIONS.keys())} [default: {', '.join(DEFAULT_FEATURES)}]"
        )
        if not text.strip(): return DEFAULT_FEATURES
        chosen = [f.strip() for f in text.split(',') if f.strip() in FEATURE_FUNCTIONS]
        return chosen or DEFAULT_FEATURES

    @Tool
    def ask_metrics(self):
        text = self.prompt(f"Select metrics to report: {', '.join(METRICS)} [default: ctr,cpc,conv_rate]")
        if not text.strip(): return ['ctr','cpc','conv_rate']
        chosen = [m.strip() for m in text.split(',') if m.strip() in METRICS]
        return chosen or ['ctr','cpc','conv_rate']

    @Tool
    def ask_weights(self, features, metrics):
        # dynamic defaults: equal weights summing to 1 for CTR features
        n = len(features)
        default_w = {f:1.0/n for f in features}  # uniform
        intercept = 0.0
        base_cpc = 1.50
        conv_factor = 0.10
        # prompt user if they want custom
        if self.confirm("Use default feature weights for CTR? (y/n) "):
            w = default_w
        else:
            w = {}
            for f in features:
                w[f] = float(self.prompt(f"Weight for CTR feature '{f}' [default {default_w[f]:.3f}]:") or default_w[f])
        intercept = float(self.prompt("CTR intercept (default 0.0):") or intercept)
        base_cpc = float(self.prompt("Base CPC USD (default 1.50):") or base_cpc)
        conv_factor = float(self.prompt("Click-to-lead conv factor (default 0.10):") or conv_factor)
        return {'feature_weights':w,'intercept':intercept,'base_cpc':base_cpc,'conv_factor':conv_factor}

    @Tool
    def load_data(self, features):
        df = pd.read_csv(CSV_PATH)
        # compute each selected feature
        for f in features:
            df[f] = FEATURE_FUNCTIONS[f](df)
        # build feature tensor
        feature_idx = {f:i for i,f in enumerate(features)}
        X = torch.tensor(df[features].values, dtype=torch.float32)
        # save for reporting
        self._df = df[['zip code tabulation area','B01002_001E','B19013_001E']].copy()
        self._X = X
        return feature_idx

    @Tool
    def simulate(self, feature_idx, weights_dict, budget, variants, metrics):
        X = self._X
        df = self._df
        # instantiate model
        model = LeadGenModel(
            feature_weights=weights_dict['feature_weights'],
            intercept=weights_dict['intercept'],
            base_cpc=weights_dict['base_cpc'],
            conv_factor=weights_dict['conv_factor'],
            feature_idx=feature_idx
        )
        ctr, cpc, conv = model(X)
        # impressions & leads
        slice_budget = budget / variants
        impr = slice_budget / cpc
        leads = impr * conv
        # compile results
        results = {}
        for i in range(variants):
            key = f'Variant_{i+1}'
            stats = { 'ctr': ctr.mean().item(), 'cpc': cpc.mean().item(), 'conv_rate': conv.mean().item(), 'impressions': int(impr.sum().item()), 'leads': int(leads.sum().item()) }
            # actionable top ZIPs
            for m in metrics:
                arr = {'ctr':ctr,'cpc':cpc,'conv_rate':conv,'impressions':impr,'leads':leads}[m]
                zip_,age,inc = get_top_zip(df, arr)
                stats[f'top_{m}_zip'] = zip_
                stats[f'top_{m}_med_age'] = age
                stats[f'top_{m}_med_inc'] = inc
            # sensitivity analysis (+10% weight bump)
            sens = {}
            for feat,w in weights_dict['feature_weights'].items():
                w_bump = weights_dict['feature_weights'].copy()
                w_bump[feat] = w * 1.1
                model_bump = LeadGenModel(w_bump, weights_dict['intercept'], weights_dict['base_cpc'], weights_dict['conv_factor'], feature_idx)
                ctr_b,_ ,_ = model_bump(X)
                sens[f'bump_{feat}'] = (ctr_b.mean().item() - stats['ctr']) / stats['ctr'] * 100
            stats['sensitivity_pct_change_ctr'] = sens
            results[key] = stats
        return results

    @Tool
    def report(self, results):
        self.say("\n=== Simulation & Insights ===")
        for variant, stats in results.items():
            self.say(f"\n{variant}:")
            for k,v in stats.items():
                self.say(f"  {k}: {v}")

    def run(self):
        self.say("Welcome to MarketingSimulator! Fully dynamic and customizable.")
        _ = self.ask_business()
        budget = self.ask_budget()
        variants = self.ask_variants()
        features = self.ask_fields()
        metrics = self.ask_metrics()
        weights_dict = self.ask_weights(features, metrics)
        feature_idx = self.load_data(features)
        results = self.simulate(feature_idx, weights_dict, budget, variants, metrics)
        self.report(results)

if __name__ == '__main__':
    run_agent(MarketingSimulator)
