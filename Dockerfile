# Use an official lightweight Python image
FROM python:3.10-slim

# 1. Set working directory
WORKDIR /app

# 2. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application code
COPY . .

# 4. Expose the port your FastMCP server listens on
EXPOSE 8000

# 5. Default command to run your server (uses PORT env var or 8000)
CMD ["sh", "-c", "fastmcp run main.py:mcp --transport sse --host 0.0.0.0 --port ${PORT:-8000}"]
