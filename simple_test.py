#!/usr/bin/env python3
"""Simple test script to interface with mcp."""
import mcp
from pprint import pprint

# Print available classes and methods in mcp
print("Available items in mcp:")
pprint(dir(mcp))

# Print more details about Tool class
print("\nDetails about mcp.Tool:")
pprint(dir(mcp.Tool))

print("\nChecking if specific classes exist:")
for class_name in ['Agent', 'Tool', 'run_agent']:
    exists = hasattr(mcp, class_name)
    print(f"  mcp.{class_name}: {'EXISTS' if exists else 'NOT FOUND'}")