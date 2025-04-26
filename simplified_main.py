#!/usr/bin/env python3
"""
Simplified marketing campaign simulator using mcp's Tool class 
without Agent and run_agent which are not found in the package.
"""
import os
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from mcp import Tool
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Path to pre-fetched ZCTA demographics CSV
CSV_PATH = Path(__file__).parent / 'data' / 'zcta_demographics.csv'
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Missing demographics CSV at {CSV_PATH}")

# Tuned regression coefficients for lead-generation modeling
ALPHAS = {
    'base_ctr':     0.01,    # baseline click-through rate (lowered)
    'age':          0.005,   # weight for %18–34 (lowered)
    'income':       0.003,   # weight for median income (lowered)
    'broadband':    0.002,   # weight for broadband penetration (lowered)
    'base_cpc':     0.80,    # base cost-per-click ($) (lowered)
    'conv_factor':  0.05,    # fraction of clicks that convert (lowered)
}

# Feature indices in the normalized feature vector
IDX = {
    'pct18_34':    3,  # column index for % 18–34
    'med_inc':     7,  # column index for median household income
    'broadband':   9,  # column index for broadband penetration
    'disp_income': 6,  # disposable income proxy (med_income / avg_household_size)
}

# Default values for non-interactive usage
DEFAULT_VALUES = {
    "business_desc": "Cloud-based accounting software for small businesses with automated tax filing",
    "budget": 2000.0,  # Reduced budget for more realistic numbers
    "variants": 3
}

# -----------------------------------------------------------------------------
# Lead-Generation Model Definition
# -----------------------------------------------------------------------------
class LeadGenModel(nn.Module):
    def __init__(self, alphas: dict):
        super().__init__()
        # regression weights
        self.a0 = alphas['base_ctr']
        self.a_age = alphas['age']
        self.a_inc = alphas['income']
        self.a_bb = alphas['broadband']
        self.base_cpc = alphas['base_cpc']
        self.conv_factor = alphas['conv_factor']

    def forward(self, x: torch.Tensor):
        # Linear combination for CTR
        raw_ctr = (
            self.a0
            + self.a_age * x[:, IDX['pct18_34']]
            + self.a_inc * x[:, IDX['med_inc']]
            + self.a_bb * x[:, IDX['broadband']]
        )
        ctr = torch.sigmoid(raw_ctr)

        # CPC inversely related to disposable income
        # Apply abs to disposable income and add epsilon to avoid division by zero
        epsilon = 1e-5
        cpc = self.base_cpc / (1.0 + torch.abs(x[:, IDX['disp_income']]) + epsilon)

        # Conversion rate is a fraction of clicks
        conv_rate = ctr * self.conv_factor

        return ctr, cpc, conv_rate

# -----------------------------------------------------------------------------
# Simplified Marketing Simulator
# -----------------------------------------------------------------------------
class SimplifiedMarketingSimulator:
    def __init__(self, use_defaults=True):
        self.business_desc = None
        self.budget = None
        self.variants = None
        self.use_defaults = use_defaults
        
    def prompt(self, message: str) -> str:
        """Prompt function to get user input or use default values"""
        if self.use_defaults:
            if "B2B service" in message:
                return DEFAULT_VALUES["business_desc"]
            elif "budget" in message:
                print(f"{message} {DEFAULT_VALUES['budget']}")
                return str(DEFAULT_VALUES["budget"])
            elif "variants" in message:
                print(f"{message} {DEFAULT_VALUES['variants']}")
                return str(DEFAULT_VALUES["variants"])
            else:
                return ""
        else:
            return input(f"{message} ")
    
    def say(self, message: str):
        """Simple function to display output"""
        print(message)
    
    def ask_business(self) -> str:
        value = self.prompt(
            "Describe your B2B service for SMBs and its unique value proposition:"
        )
        print(f"Business Description: {value}")
        return value

    def ask_budget(self) -> float:
        return float(self.prompt("Monthly ad budget (USD)?"))

    def ask_variants(self) -> int:
        k = int(self.prompt("How many campaign variants would you like to test? (3–5)"))
        return max(3, min(k, 5))

    def load_census(self) -> torch.Tensor:
        """Load census data and convert to tensor"""
        # Load the pre-fetched ZCTA demographic CSV
        self.say("Loading demographic data...")
        df = pd.read_csv(CSV_PATH, header=None)
        
        # The CSV doesn't have headers, but we can assign our own based on the indices
        # Most important: col_3 is age %, col_7 is median income, col_32 is broadband %
        
        # Convert dataframe to numeric where possible
        for col in df.columns:
            if col > 0:  # Skip the first column which is ZCTA ID
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extract features for our model
        # We're using specific columns based on the IDX mapping
        # col_3: % 18-34 (age), col_7: median income, col_32: broadband %
        # col_27: household size for disposable income calculation
        features_df = df[[3, 7, 27, 32]].copy()
        
        # Replace NaN values with column means
        features_df = features_df.fillna(features_df.mean())
        
        # Normalize the features
        for col in features_df.columns:
            col_data = features_df[col]
            features_df[col] = (col_data - col_data.mean()) / col_data.std()
        
        # Calculate disposable income as median income / household size
        features_df['disp_income'] = features_df[7] / features_df[27]
        
        # Create a tensor with the features in the order expected by our model
        # Re-order columns to match the IDX mapping
        feature_tensor = torch.zeros((len(features_df), 10), dtype=torch.float32)
        feature_tensor[:, IDX['pct18_34']] = torch.tensor(features_df[3].values, dtype=torch.float32)
        feature_tensor[:, IDX['med_inc']] = torch.tensor(features_df[7].values, dtype=torch.float32)
        feature_tensor[:, IDX['disp_income']] = torch.tensor(features_df['disp_income'].values, dtype=torch.float32)
        feature_tensor[:, IDX['broadband']] = torch.tensor(features_df[32].values, dtype=torch.float32)
        
        # For realistic numbers, let's only use a small subset of the data
        # Use only 100 random rows - this is a demo after all
        idx = torch.randperm(len(feature_tensor))[:100]
        
        return feature_tensor[idx]

    def simulate(self, features: torch.Tensor, variants: int, budget: float) -> Dict[str, Dict[str, Any]]:
        """Run the marketing simulation"""
        # Initialize the lead-gen model
        model = LeadGenModel(ALPHAS)
        ctrs, cpcs, convs = model(features)

        # Allocate budget equally across variants
        slice_budget = budget / variants
        results = {}

        # Variant adjustment factors (to make variants different)
        # Each variant will have slightly different performance
        var_factors = {
            "ctr": [1.0, 0.95, 1.05],
            "cpc": [1.0, 1.05, 0.95],
            "conv": [1.0, 0.97, 1.03]
        }

        for i in range(variants):
            # Apply variant-specific adjustment factors
            ctr_factor = var_factors["ctr"][i] if i < len(var_factors["ctr"]) else 1.0
            cpc_factor = var_factors["cpc"][i] if i < len(var_factors["cpc"]) else 1.0
            conv_factor = var_factors["conv"][i] if i < len(var_factors["conv"]) else 1.0
            
            variant_ctr = ctrs * ctr_factor
            variant_cpc = cpcs * cpc_factor
            variant_conv = convs * conv_factor
            
            impr = slice_budget / variant_cpc
            leads = impr * variant_conv
            
            results[f"Variant_{i+1}"] = {
                'ctr':          variant_ctr.mean().item(),
                'cpc':          variant_cpc.mean().item(),
                'conv_rate':    variant_conv.mean().item(),
                'impressions':  int(impr.sum().item()),
                'leads':        int(leads.sum().item()),
            }
        return results

    def report(self, results: Dict[str, Dict[str, Any]]):
        """Display simulation results"""
        self.say("\n=== Simulation Results ===")
        for name, m in results.items():
            self.say(
                f"{name}: CTR={m['ctr']:.2%}, CPC=${m['cpc']:.2f}, "
                f"ConvRate={m['conv_rate']:.2%}, Impr={m['impressions']:,}, "
                f"Leads={m['leads']:,}"
            )

    def run(self):
        """Main execution flow"""
        # Agent conversation flow
        self.say("Welcome to the Marketing Simulator! 🎯")
        self.business_desc = self.ask_business()
        self.budget = self.ask_budget()
        self.variants = self.ask_variants()
        features = self.load_census()

        # Run simulation and report
        results = self.simulate(features, self.variants, self.budget)
        self.report(results)

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    simulator = SimplifiedMarketingSimulator(use_defaults=True)
    simulator.run()