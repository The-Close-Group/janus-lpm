#!/usr/bin/env python3
"""
MarketingSimulator FastMCP Server - Production

A production-grade entrypoint that composes:
  - FeatureSelectionService (feat_)
  - ModelService           (model_)
  - SimulationService      (sim_)
  - AgentTorchService      (atm_)

**Interactive Client Example:**
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
        print(train, back)

        # Run batch simulation
        sim = await client.call_tool(
            "sim_simulate", {
                "data": {"features": synth['X']},
                "weights": {"feature_weights": {f:1.0/3 for f in feats[:3]}},
                "intercept":0.0, "base_cpc":1.5, "conv_factor":0.1,
                "variants":3, "budget":10000, "metrics":["ctr","leads"]
            }
        )
        print(sim)

        # Optimize budget allocation
        opt = await client.call_tool(
            "sim_optimize_budget_allocation", {
                "features": feats[:3],
                "total_budget": 10000.0,
                "num_iterations": 5
            }
        )
        print(opt)

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

        # Run distributed campaign
        dc = await client.call_tool(
            "atm_run_distributed_campaign", {
                "features": feats[:3],
                "model_size": "medium",
                "checkpoint_path": train['checkpoint_path'],
                "total_budget": 10000.0,
                "num_campaigns": 3
            }
        )
        print(dc)

# To run example: asyncio.run(example_workflow())
```

"""
import json
import asyncio
import argparse
from fastmcp import FastMCP
from fastmcp.resources import TextResource

# -----------------------------------------------------------------------------
# AgentCard
# -----------------------------------------------------------------------------
AGENT_CARD = {
    "name":        "MarketingSimulator",
    "description": "Production FB/IG lead-gen simulation pipeline with AgentTorch integration",
    "version":     "2.2.0",
    "input_modes": ["text"],
    "output_modes":["text","data"],
    "capabilities": {"streaming": True, "pushNotifications": False}
}

# -----------------------------------------------------------------------------
# FastMCP App
# -----------------------------------------------------------------------------
mcp = FastMCP(
    name="MarketingSimulator",
    dependencies=["pandas", "torch", "scikit-learn", "agenttorch", "numpy"]
)

# AgentCard resource
tc = TextResource(
    uri="resource://agentcard",
    name="AgentCard",
    text=json.dumps(AGENT_CARD),
    mime_type="application/json"
)
mcp.add_resource(tc)

# Health check endpoint
@mcp.resource("resource://*/health")
def health() -> str:
    return "OK"

# -----------------------------------------------------------------------------
# Mount Subservers
# -----------------------------------------------------------------------------
from server.feature_selection import app as feature_app
from server.model import app as model_app
from server.simulation_service import app as sim_app
from server.agent_service import app as agent_app

async def setup_subservers():
    await mcp.import_server(prefix="feat", app=feature_app)
    await mcp.import_server(prefix="model", app=model_app)
    await mcp.import_server(prefix="sim", app=sim_app)
    await mcp.import_server(prefix="atm", app=agent_app)

# Preload subservers
asyncio.run(setup_subservers())

# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run MarketingSimulator FastMCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mcp.run(transport="sse", host=args.host, port=args.port)