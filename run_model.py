#!/usr/bin/env python3
"""
Script to train a model using the existing components and run inference.
"""
import os
import sys
import time
import json
import pandas as pd
import numpy as np
from fastmcp import Context

# Add the current directory to the path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import server components
from server.feature_selection import load_data, list_features, select_features, generate_synthetic_targets
from server.model import train_model, backtest_model, MODEL_CONFIGS, LeadGenNet
from server.agent_service import run_population_model, run_distributed_campaign

def main():
    # Step 1: Load data using the existing function
    print("\n= Step 1: Loading data =")
    result = load_data()
    print(result)
    
    # Step 2: List available features
    print("\n= Step 2: Getting available features =")
    features = list_features()
    selected_features = features[:5]  # Start with the first 5 features
    print(f"Available features: {len(features)}")
    print(f"Selected features for training: {selected_features}")
    
    # Step 3: Select features
    print("\n= Step 3: Computing selected features =")
    feature_data = select_features(selected_features)
    print(f"Feature computation complete: {len(feature_data)} features computed")
    
    # Step 4: Generate synthetic targets for training
    print("\n= Step 4: Generating synthetic targets =")
    start_time = time.time()
    synth_data = generate_synthetic_targets(selected_features)
    print(f"Synthetic data generation complete in {time.time() - start_time:.2f}s")
    print(f"Generated {len(synth_data['X'])} samples")
    
    # Step 5: Train the model
    print("\n= Step 5: Training model =")
    model_size = "medium"
    start_time = time.time()
    train_result = train_model(model_size, selected_features, epochs=15)
    print(f"Model training complete in {time.time() - start_time:.2f}s")
    print(f"Training metrics: RMSE CTR: {train_result['rmse_ctr']:.4f}, RMSE Conv: {train_result['rmse_conv']:.4f}")
    print(f"Model checkpoint saved to: {train_result['checkpoint_path']}")
    
    # Step 6: Backtest the model
    print("\n= Step 6: Backtesting model =")
    backtest_result = backtest_model(model_size)
    print(f"Backtest results: R2 CTR: {backtest_result['r2_ctr']:.4f}, MAE CPC: {backtest_result['mae_cpc']:.4f}")
    
    # Step 7: Run population model (AgentTorch integration)
    print("\n= Step 7: Running AgentTorch population model =")
    start_time = time.time()
    population_result = run_population_model(
        features=selected_features,
        model_size=model_size,
        checkpoint_path=train_result['checkpoint_path'],
        global_budget=10000.0,
        conv_factor=0.1,
        num_steps=3
    )
    print(f"Population simulation complete in {time.time() - start_time:.2f}s")
    print(f"Population size: {population_result['population_size']}")
    print(f"Total conversions: {population_result['total_conversions']}")
    print(f"Overall ROI: {population_result['overall_roi']:.2f}x")
    
    # Step 8: Run distributed campaign
    print("\n= Step 8: Running distributed campaign simulation =")
    start_time = time.time()
    campaign_result = run_distributed_campaign(
        features=selected_features,
        model_size=model_size,
        checkpoint_path=train_result['checkpoint_path'],
        total_budget=10000.0,
        num_campaigns=3
    )
    print(f"Distributed campaign simulation complete in {time.time() - start_time:.2f}s")
    print(f"Best campaign: {campaign_result['best_campaign']} with ROI: {campaign_result['best_campaign_roi']:.2f}x")
    print(f"Total conversions across all campaigns: {campaign_result['total_conversions']}")
    
    # Step 9: Save results to file
    print("\n= Step 9: Saving results =")
    results = {
        "training": train_result,
        "backtest": backtest_result,
        "population_model": population_result,
        "distributed_campaign": campaign_result
    }
    
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to model_results.json")
    
    print("\n= Complete! =")

if __name__ == "__main__":
    main()