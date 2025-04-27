import numpy as np
import pandas as pd
from typing import Dict, List, Any
from fastmcp import FastMCP, Context

from server.model import LeadGenNet
from server.feature_selection import load_data, select_features, generate_synthetic_targets

# FastMCP subserver for simulation
app = FastMCP(name="SimulationService")

@app.tool()
def simulate(
    data: Dict[str, Any],
    weights: Dict[str, Any],
    intercept: float = 0.0,
    base_cpc: float = 1.5,
    conv_factor: float = 0.1,
    variants: int = 1,
    budget: float = 10000.0,
    metrics: List[str] = ['ctr', 'cpc', 'conv', 'roi']
) -> Dict[str, Any]:
    """
    Simulate ad campaign results using weighted features and synthetic targets.
    
    Args:
        data: Dictionary with feature data (features, zip_codes, etc.)
        weights: Dictionary with feature weights
        intercept: Intercept value for the linear model
        base_cpc: Base cost per click
        conv_factor: Conversion factor
        variants: Number of variants to simulate
        budget: Total budget
        metrics: List of metrics to return
        
    Returns:
        Dictionary with simulation results for each variant
    """
    features = data.get('features', [])
    if not features:
        raise ValueError("No feature data provided")
    
    feature_weights = weights.get('feature_weights', {})
    if not feature_weights:
        raise ValueError("No feature weights provided")
    
    # Get feature values
    X = np.array(features)
    
    # Create weighted sum
    weighted_sum = 0
    for feat, weight in feature_weights.items():
        if feat in data:
            weighted_sum += weight * np.array(data[feat])
    weighted_sum += intercept
    
    # Generate CTR, CPC, and conversion rates
    ctr_base = 1 / (1 + np.exp(-weighted_sum * 0.5))  # Sigmoid
    cpc_base = base_cpc * (1 + 0.2 * weighted_sum)    # Linear with base
    conv_base = conv_factor * ctr_base                # Proportional to CTR
    
    # Simulate variants
    variant_results = []
    for v in range(variants):
        # Add random variation
        ctr = ctr_base * np.random.uniform(0.8, 1.2)
        cpc = cpc_base * np.random.uniform(0.9, 1.1)
        conv = conv_base * np.random.uniform(0.9, 1.1)
        
        # Clip to valid ranges
        ctr = np.clip(ctr, 0.001, 0.3)
        cpc = np.clip(cpc, 0.5, 5.0)
        conv = np.clip(conv, 0.0001, 0.2)
        
        # Calculate campaign metrics
        spent = budget / variants
        impressions = int(spent / np.mean(cpc) * 1000)
        clicks = int(impressions * np.mean(ctr))
        conversions = int(clicks * np.mean(conv))
        cost_per_lead = spent / conversions if conversions > 0 else float('inf')
        revenue = conversions * 100  # Assume $100 per conversion
        roi = (revenue - spent) / spent if spent > 0 else 0
        
        # Compile results
        result = {
            'variant': v + 1,
            'budget': spent,
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'ctr': float(np.mean(ctr)),
            'cpc': float(np.mean(cpc)),
            'conv_rate': float(np.mean(conv)),
            'cost_per_lead': float(cost_per_lead),
            'revenue': float(revenue),
            'roi': float(roi)
        }
        
        # Filter requested metrics
        if metrics:
            result = {k: v for k, v in result.items() if k in metrics or k == 'variant'}
        
        variant_results.append(result)
    
    return {
        'variants': variants,
        'total_budget': budget,
        'results': variant_results
    }

@app.tool()
def compare_strategies(
    features: List[str],
    strategies: List[Dict[str, Any]],
    budget_per_strategy: float = 5000.0,
    conv_value: float = 100.0
) -> Dict[str, Any]:
    """
    Compare multiple campaign strategies using the same feature set.
    
    Args:
        features: List of feature names
        strategies: List of strategy dictionaries with weights and parameters
        budget_per_strategy: Budget to allocate per strategy
        conv_value: Value of each conversion ($)
        
    Returns:
        Dictionary comparing strategy results
    """
    # Ensure data is loaded
    load_data()
    
    # Generate synthetic data if not already done
    if Context.storage.get('synthetic_data') is None:
        generate_synthetic_targets(features)
    
    synth_data = Context.storage.get('synthetic_data')
    X = synth_data['X']
    
    # Run each strategy
    strategy_results = []
    for i, strategy in enumerate(strategies):
        name = strategy.get('name', f"Strategy {i+1}")
        weights = strategy.get('weights', {f: 1.0/len(features) for f in features})
        base_cpc = strategy.get('base_cpc', 1.5)
        
        # Create feature data dict
        data = {'features': X}
        for j, feat in enumerate(features):
            data[feat] = [row[j] for row in X]
        
        # Simulate
        sim_result = simulate(
            data=data,
            weights={'feature_weights': weights},
            intercept=strategy.get('intercept', 0.0),
            base_cpc=base_cpc,
            conv_factor=strategy.get('conv_factor', 0.1),
            variants=1,
            budget=budget_per_strategy,
            metrics=['ctr', 'cpc', 'conv_rate', 'clicks', 'conversions', 'roi']
        )
        
        variant = sim_result['results'][0]
        
        # Calculate additional metrics
        spend = budget_per_strategy
        conversions = variant['conversions']
        revenue = conversions * conv_value
        roi = (revenue - spend) / spend if spend > 0 else 0
        cpl = spend / conversions if conversions > 0 else float('inf')
        
        strategy_results.append({
            'name': name,
            'spend': spend,
            'clicks': variant['clicks'],
            'conversions': conversions,
            'ctr': variant['ctr'],
            'conv_rate': variant['conv_rate'],
            'revenue': revenue,
            'roi': roi,
            'cpl': cpl
        })
    
    # Find best and worst strategies
    best_strategy = max(strategy_results, key=lambda x: x['roi'])
    worst_strategy = min(strategy_results, key=lambda x: x['roi'])
    
    return {
        'strategies_compared': len(strategies),
        'total_budget': budget_per_strategy * len(strategies),
        'best_strategy': best_strategy['name'],
        'best_strategy_roi': best_strategy['roi'],
        'worst_strategy': worst_strategy['name'],
        'worst_strategy_roi': worst_strategy['roi'],
        'strategy_results': strategy_results
    }

@app.tool()
def optimize_budget_allocation(
    features: List[str],
    total_budget: float = 10000.0,
    num_iterations: int = 5,
    conv_value: float = 100.0
) -> Dict[str, Any]:
    """
    Use a simple evolutionary algorithm to optimize budget allocation across features.
    
    Args:
        features: List of feature names to optimize weights for
        total_budget: Total campaign budget
        num_iterations: Number of optimization iterations
        conv_value: Value of each conversion ($)
        
    Returns:
        Dictionary with optimization results
    """
    # Ensure data is loaded
    load_data()
    
    # Generate synthetic data if not already done
    if Context.storage.get('synthetic_data') is None:
        generate_synthetic_targets(features)
    
    synth_data = Context.storage.get('synthetic_data')
    X = synth_data['X']
    
    # Initial random weights
    num_features = len(features)
    population_size = 5
    population = []
    
    # Create initial population
    for i in range(population_size):
        # Random weights that sum to 1
        weights = np.random.random(num_features)
        weights = weights / np.sum(weights)
        
        # Create strategy
        strategy = {
            'name': f"Generation 0, Individual {i+1}",
            'weights': {features[j]: weights[j] for j in range(num_features)},
            'base_cpc': np.random.uniform(0.8, 2.0),
            'conv_factor': np.random.uniform(0.05, 0.15),
            'intercept': np.random.uniform(-0.5, 0.5)
        }
        
        population.append(strategy)
    
    # Run optimization iterations
    best_strategies = []
    
    for gen in range(num_iterations):
        # Evaluate current population
        results = compare_strategies(
            features=features,
            strategies=population,
            budget_per_strategy=total_budget / population_size,
            conv_value=conv_value
        )
        
        # Get the best strategy from this generation
        strategy_results = results['strategy_results']
        sorted_strategies = sorted(strategy_results, key=lambda x: x['roi'], reverse=True)
        best_strategy = sorted_strategies[0]
        best_strategies.append({
            'generation': gen,
            'strategy_name': best_strategy['name'],
            'roi': best_strategy['roi'],
            'conversions': best_strategy['conversions'],
            'cpl': best_strategy['cpl']
        })
        
        # Create next generation through selection and mutation
        next_population = []
        
        # Always keep the best strategy
        best_idx = [i for i, s in enumerate(strategy_results) if s['name'] == best_strategy['name']][0]
        next_population.append(population[best_idx])
        
        # Create new strategies through mutation and crossover
        for i in range(population_size - 1):
            # Select parent (biased toward better performers)
            parent_idx = np.random.choice(
                range(population_size),
                p=np.array([max(0.1, s['roi']) for s in strategy_results]) / 
                  sum(max(0.1, s['roi']) for s in strategy_results)
            )
            parent = population[parent_idx]
            
            # Mutate weights
            new_weights = {
                f: parent['weights'][f] * np.random.uniform(0.8, 1.2) 
                for f in features
            }
            # Normalize to sum to 1
            weight_sum = sum(new_weights.values())
            new_weights = {f: w / weight_sum for f, w in new_weights.items()}
            
            # Create new strategy
            new_strategy = {
                'name': f"Generation {gen+1}, Individual {i+1}",
                'weights': new_weights,
                'base_cpc': parent['base_cpc'] * np.random.uniform(0.9, 1.1),
                'conv_factor': parent['conv_factor'] * np.random.uniform(0.9, 1.1),
                'intercept': parent['intercept'] + np.random.uniform(-0.1, 0.1)
            }
            
            next_population.append(new_strategy)
        
        # Update population for next generation
        population = next_population
    
    # Final evaluation
    final_results = compare_strategies(
        features=features,
        strategies=population,
        budget_per_strategy=total_budget / population_size,
        conv_value=conv_value
    )
    
    # Get the overall best strategy
    all_best = max(best_strategies, key=lambda x: x['roi'])
    
    # Format weights of best strategy for output
    best_idx = [i for i, s in enumerate(best_strategies) if s['generation'] == all_best['generation']][0]
    best_weights = population[best_idx]['weights']
    
    return {
        'num_iterations': num_iterations,
        'total_budget': total_budget,
        'conv_value': conv_value,
        'best_generation': all_best['generation'],
        'best_roi': all_best['roi'],
        'best_conversions': all_best['conversions'],
        'best_cpl': all_best['cpl'],
        'optimized_weights': best_weights,
        'optimization_history': best_strategies
    }

# To serve standalone:
if __name__ == '__main__':
    app.run(transport='sse', host='0.0.0.0', port=8004)