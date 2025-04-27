# Dockerfile for MarketingSimulator (auto-clone from GitHub)

# 1. Base image
FROM python:3.10-slim

# 2. Install Git (to clone the repo) and clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# 3. Clone the repository into /app
WORKDIR /app
RUN git clone https://github.com/The-Close-Group/janus-lpm.git .

# 4. Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose FastMCP port
EXPOSE 8000

# 6. Entrypoint: run the FastMCP server (SSE transport)
CMD ["sh", "-c", \
    "fastmcp run main.py:mcp --transport sse --host 0.0.0.0 --port ${PORT:-8000}"\
]
