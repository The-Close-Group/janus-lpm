# Dockerfile for MarketingSimulator with AgentTorch integration

# 1. Base image
FROM python:3.10-slim

# 2. Install system dependencies and clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements first to leverage Docker caching
COPY requirements.txt ./

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the application code
COPY . .

# 7. Create directories for model checkpoints and data
RUN mkdir -p /app/checkpoints /app/data

# 8. Expose FastMCP port
EXPOSE 8000

# 9. Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# 10. Entrypoint: run the FastMCP server (SSE transport)
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]