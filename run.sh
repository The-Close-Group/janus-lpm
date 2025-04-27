#!/bin/bash

# Set up the virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    pip install hypercorn
    touch venv/.installed
fi

# Check if the ZCTA demographics file exists
if [ ! -f "data/zcta_demographics.csv" ]; then
    echo "Downloading ZCTA demographics data..."
    bash data/fetch_zcta_demographics.sh
fi

# Start the server with Hypercorn
echo "Starting the server with Hypercorn..."
fastmcp run main.py:mcp --transport sse --host 0.0.0.0 

# Use below for original FastMCP server without Hypercorn
# python main.py

# Use below for simplified version
# python simplified_main.py