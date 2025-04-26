#!/usr/bin/env python3
"""Test script to examine agent_torch."""
import agent_torch
from pprint import pprint

# Print available classes and methods in agent_torch
print("Available items in agent_torch:")
pprint(dir(agent_torch))

# Check if Agent and run_agent are in agent_torch
print("\nChecking if specific classes exist in agent_torch:")
for class_name in ['Agent', 'run_agent']:
    exists = hasattr(agent_torch, class_name)
    print(f"  agent_torch.{class_name}: {'EXISTS' if exists else 'NOT FOUND'}")

# Check in agent_torch submodules
print("\nChecking submodules for Agent class:")
submodules = [name for name in dir(agent_torch) if not name.startswith('_') and not callable(getattr(agent_torch, name))]
for submodule_name in submodules:
    try:
        submodule = getattr(agent_torch, submodule_name)
        agent_exists = hasattr(submodule, 'Agent')
        if agent_exists:
            print(f"  agent_torch.{submodule_name}.Agent: EXISTS")
    except Exception as e:
        print(f"  Error checking {submodule_name}: {e}")