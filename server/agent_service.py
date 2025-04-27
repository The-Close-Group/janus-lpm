import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any
from fastmcp import FastMCP, Context

from server.model import LeadGenNet
from server.feature_selection import load_data, select_features, generate_synthetic_targets

# FastMCP subserver for AgentTorch integration
app = FastMCP(name="AgentTorchService")

class PopulationModel:
    """
    AgentTorch-compatible population model that simulates lead generation
    across a population of potential customers.
    """
    def __init__(self, features, model_size, checkpoint_path, 
                 global_budget=10000.0, conv_factor=0.1):
        self.features = features
        self.model_size = model_size
        self.checkpoint_path = checkpoint_path
        self.global_budget = global_budget
        self.conv_factor = conv_factor
        self.load_model()
        
    def load_model(self):
        # Load model configuration
        from server.model import MODEL_CONFIGS
        cfg = MODEL_CONFIGS.get(self.model_size)
        if cfg is None:
            raise ValueError(f"Unknown model size: {self.model_size}")
        
        # Create model
        input_dim = len(self.features)
        self.model = LeadGenNet(input_dim=input_dim, hidden_dim=cfg['hidden_dim'])
        
        # Load checkpoint
        try:
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model checkpoint: {e}")
            
    def simulate_step(self, feature_values, budget_allocation):
        """
        Simulate a single step of the lead generation process.
        
        Args:
            feature_values: Dictionary mapping feature names to values
            budget_allocation: Dictionary with marketing budget allocation
            
        Returns:
            Dictionary with simulation results
        """
        # Prepare inputs
        X = np.array([[feature_values[f] for f in self.features]], dtype=np.float32)
        X_tensor = torch.from_numpy(X)
        
        # Run model prediction
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()[0]
        
        ctr = predictions[0]
        cpc = predictions[1]
        conv_rate = predictions[2]
        
        # Calculate metrics
        spent = min(budget_allocation.get('budget', 0), self.global_budget)
        impressions = int(spent / cpc) if cpc > 0 else 0
        clicks = int(impressions * ctr)
        conversions = int(clicks * conv_rate * self.conv_factor)
        
        return {
            'ctr': float(ctr),
            'cpc': float(cpc),
            'conv_rate': float(conv_rate),
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'spent': spent,
            'revenue': conversions * spent * 2,  # Just an example revenue calculation
            'roi': (conversions * spent * 2) / spent if spent > 0 else 0
        }

@app.tool()
def run_population_model(
    features: List[str],
    model_size: str = "medium",
    checkpoint_path: str = None,
    global_budget: float = 10000.0,
    conv_factor: float = 0.1,
    num_steps: int = 5
) -> Dict[str, Any]:
    """
    Run a multi-step simulation using AgentTorch population model approach.
    
    Args:
        features: List of feature names to use
        model_size: Model size (small, medium, large, huge)
        checkpoint_path: Path to the model checkpoint
        global_budget: Global marketing budget
        conv_factor: Conversion factor
        num_steps: Number of simulation steps
        
    Returns:
        Dictionary with simulation results
    """
    # Ensure data is loaded
    load_data()
    
    # Select features and generate synthetic targets if not already done
    if Context.storage.get('synthetic_data') is None:
        generate_synthetic_targets(features)
    
    # If no checkpoint_path provided, use default
    if checkpoint_path is None:
        checkpoint_path = f"model_{model_size}_best.pt"
    
    # Create population model
    model = PopulationModel(
        features=features,
        model_size=model_size,
        checkpoint_path=checkpoint_path,
        global_budget=global_budget,
        conv_factor=conv_factor
    )
    
    # Get synthetic data
    synth_data = Context.storage.get('synthetic_data')
    feature_matrix = synth_data['X']
    
    # Prepare population
    population_size = min(100, len(feature_matrix))
    population_indices = np.random.choice(len(feature_matrix), size=population_size, replace=False)
    
    # Run simulation steps
    results = []
    total_metrics = {
        'impressions': 0,
        'clicks': 0,
        'conversions': 0,
        'spent': 0,
        'revenue': 0
    }
    
    for step in range(num_steps):
        step_metrics = {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'spent': 0,
            'revenue': 0
        }
        
        step_results = []
        for i in population_indices:
            # Get feature values for this population member
            feature_values = {feat: feature_matrix[i][j] for j, feat in enumerate(features)}
            
            # Determine budget allocation (simple strategy for now)
            budget_allocation = {
                'budget': global_budget / (population_size * num_steps),
                'feature_weights': {f: 1.0/len(features) for f in features}
            }
            
            # Run simulation step
            result = model.simulate_step(feature_values, budget_allocation)
            step_results.append(result)
            
            # Update step metrics
            for k in step_metrics:
                step_metrics[k] += result.get(k, 0)
        
        # Calculate averages
        step_summary = {
            'step': step + 1,
            'metrics': {
                k: v / population_size for k, v in step_metrics.items()
            },
            'totals': step_metrics,
            'population_size': population_size
        }
        
        # Update total metrics
        for k in total_metrics:
            total_metrics[k] += step_metrics.get(k, 0)
        
        # Add to results
        results.append(step_summary)
    
    # Calculate overall ROI
    roi = total_metrics['revenue'] / total_metrics['spent'] if total_metrics['spent'] > 0 else 0
    
    return {
        'num_steps': num_steps,
        'population_size': population_size,
        'total_impressions': total_metrics['impressions'],
        'total_clicks': total_metrics['clicks'],
        'total_conversions': total_metrics['conversions'],
        'total_spent': total_metrics['spent'],
        'total_revenue': total_metrics['revenue'],
        'overall_roi': roi,
        'step_results': results
    }

@app.tool()
def run_distributed_campaign(
    features: List[str],
    model_size: str = "medium",
    checkpoint_path: str = None,
    total_budget: float = 10000.0,
    num_campaigns: int = 3,
    budget_distribution: List[float] = None
) -> Dict[str, Any]:
    """
    Run a distributed campaign simulation where budget is allocated across
    multiple independent campaigns.
    
    Args:
        features: List of feature names to use
        model_size: Model size (small, medium, large, huge) 
        checkpoint_path: Path to the model checkpoint
        total_budget: Total marketing budget
        num_campaigns: Number of campaigns to simulate
        budget_distribution: List of budget fractions for each campaign (must sum to 1.0)
        
    Returns:
        Dictionary with campaign results and comparisons
    """
    # Ensure data is loaded
    load_data()
    
    # Generate synthetic data if not already done
    if Context.storage.get('synthetic_data') is None:
        generate_synthetic_targets(features)
    
    # If no checkpoint_path provided, use default
    if checkpoint_path is None:
        checkpoint_path = f"model_{model_size}_best.pt"
    
    # Set default budget distribution if not provided
    if budget_distribution is None:
        # Equal distribution
        budget_distribution = [1.0 / num_campaigns] * num_campaigns
    elif len(budget_distribution) != num_campaigns:
        raise ValueError(f"Budget distribution must have {num_campaigns} elements")
    elif abs(sum(budget_distribution) - 1.0) > 0.0001:
        raise ValueError("Budget distribution must sum to 1.0")
    
    # Calculate campaign budgets
    campaign_budgets = [total_budget * frac for frac in budget_distribution]
    
    # Run each campaign
    campaign_results = []
    for i in range(num_campaigns):
        campaign_name = f"Campaign {i+1}"
        budget = campaign_budgets[i]
        
        # Create population model
        model = PopulationModel(
            features=features,
            model_size=model_size,
            checkpoint_path=checkpoint_path,
            global_budget=budget,
            conv_factor=0.1  # Fixed conversion factor for simplicity
        )
        
        # Get synthetic data
        synth_data = Context.storage.get('synthetic_data')
        feature_matrix = synth_data['X']
        
        # Prepare population (different for each campaign)
        population_size = min(50, len(feature_matrix))
        population_indices = np.random.choice(
            len(feature_matrix), 
            size=population_size, 
            replace=False
        )
        
        # Run campaign (single step)
        campaign_metrics = {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'spent': 0,
            'revenue': 0
        }
        
        for idx in population_indices:
            # Get feature values
            feature_values = {feat: feature_matrix[idx][j] for j, feat in enumerate(features)}
            
            # Simple budget allocation
            budget_per_member = budget / population_size
            budget_allocation = {
                'budget': budget_per_member,
                'feature_weights': {f: 1.0/len(features) for f in features}
            }
            
            # Simulate
            result = model.simulate_step(feature_values, budget_allocation)
            
            # Update campaign metrics
            for k in campaign_metrics:
                campaign_metrics[k] += result.get(k, 0)
        
        # Calculate campaign ROI
        roi = campaign_metrics['revenue'] / campaign_metrics['spent'] if campaign_metrics['spent'] > 0 else 0
        
        # Add to results
        campaign_results.append({
            'name': campaign_name,
            'budget': budget,
            'budget_fraction': budget_distribution[i],
            'population_size': population_size,
            'impressions': campaign_metrics['impressions'],
            'clicks': campaign_metrics['clicks'],
            'conversions': campaign_metrics['conversions'],
            'spent': campaign_metrics['spent'],
            'revenue': campaign_metrics['revenue'],
            'roi': roi
        })
    
    # Calculate best and worst campaigns
    best_campaign = max(campaign_results, key=lambda x: x['roi'])
    worst_campaign = min(campaign_results, key=lambda x: x['roi'])
    
    # Calculate totals
    total_metrics = {
        'impressions': sum(c['impressions'] for c in campaign_results),
        'clicks': sum(c['clicks'] for c in campaign_results),
        'conversions': sum(c['conversions'] for c in campaign_results),
        'spent': sum(c['spent'] for c in campaign_results),
        'revenue': sum(c['revenue'] for c in campaign_results),
    }
    overall_roi = total_metrics['revenue'] / total_metrics['spent'] if total_metrics['spent'] > 0 else 0
    
    return {
        'num_campaigns': num_campaigns,
        'total_budget': total_budget,
        'total_impressions': total_metrics['impressions'],
        'total_clicks': total_metrics['clicks'],
        'total_conversions': total_metrics['conversions'],
        'total_spent': total_metrics['spent'],
        'total_revenue': total_metrics['revenue'],
        'overall_roi': overall_roi,
        'best_campaign': best_campaign['name'],
        'best_campaign_roi': best_campaign['roi'],
        'worst_campaign': worst_campaign['name'],
        'worst_campaign_roi': worst_campaign['roi'],
        'campaign_results': campaign_results
    }

# To test standalone:
if __name__ == '__main__':
    app.run(transport='sse', host='0.0.0.0', port=8003)