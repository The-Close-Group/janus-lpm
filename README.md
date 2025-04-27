# MarketingSimulator with AgentTorch Integration

A production-grade marketing simulation platform built with FastMCP that integrates lead generation prediction models with AgentTorch for advanced marketing simulation and optimization.

## Features

- **Feature Selection**: Dynamic feature processing from census data
- **Model Training**: Multi-size neural network models for CTR, CPC, and conversion prediction
- **Simulation**: Marketing campaign simulation with budget and parameter optimization
- **AgentTorch Integration**: Population-based agent simulation for multi-step marketing scenarios
- **Distributed Campaigns**: Test multiple campaign strategies in parallel

## Architecture

The system is composed of four integrated services:

1. **Feature Selection Service** (`feat_`): Loads and processes census data, computes derived features
2. **Model Service** (`model_`): Trains and evaluates neural network models in various sizes 
3. **Simulation Service** (`sim_`): Runs marketing simulations and optimization algorithms
4. **AgentTorch Service** (`atm_`): Advanced agent-based simulations using population models

## Usage Example

```python
from fastmcp import Client
import asyncio

async def example_workflow():
    async with Client(mcp) as client:
        # Load census data once
        await client.call_tool("feat_load_data")

        # List and select features
        feats = await client.call_tool("feat_list_features")
        selected = await client.call_tool("feat_select_features", {"features": feats[:3]})

        # Generate synthetic targets
        synth = await client.call_tool("feat_generate_synthetic_targets", {"features": feats[:3]})

        # Train and backtest a medium model
        train = await client.call_tool("model_train_model", {"size": "medium", "features": feats[:3], "epochs": 5})
        back = await client.call_tool("model_backtest_model", {"size": "medium"})

        # Run AgentTorch multi-step model
        atm = await client.call_tool(
            "atm_run_population_model", {
                "features": feats[:3],
                "model_size": "medium",
                "checkpoint_path": train['checkpoint_path'],
                "global_budget": 10000.0,
                "conv_factor": 0.1,
                "num_steps": 5
            }
        )
        print(atm)
```

## Installation

### Local Development

1. Clone the repository:
   ```
   git clone https://github.com/your-username/marketing-simulator.git
   cd marketing-simulator
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the server:
   ```
   python main.py
   ```

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t marketing-simulator .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 marketing-simulator
   ```

## API Documentation

The FastMCP server exposes tools under these prefixes:

- **feat_**: Feature selection tools
  - `feat_load_data()` - Load census data
  - `feat_list_features()` - List available features
  - `feat_select_features(features)` - Compute selected features
  - `feat_generate_synthetic_targets(features)` - Generate target variables
  
- **model_**: Model training and evaluation
  - `model_list_models()` - List available model sizes
  - `model_build_model(size, input_dim)` - Initialize model architecture
  - `model_train_model(size, features, epochs)` - Train a model
  - `model_backtest_model(size)` - Evaluate trained model
  
- **sim_**: Simulation tools
  - `sim_simulate(data, weights, budget)` - Run marketing simulation
  - `sim_compare_strategies(features, strategies)` - Compare campaign strategies
  - `sim_optimize_budget_allocation(features, total_budget)` - Optimize budget distribution

- **atm_**: AgentTorch integration
  - `atm_run_population_model(features, model_size, global_budget)` - Multi-step simulation
  - `atm_run_distributed_campaign(features, model_size, total_budget)` - Multiple campaign simulation

## License

MIT