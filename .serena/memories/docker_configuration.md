# Docker Configuration for MCP-Logic

This document outlines the Docker configuration for the MCP-Logic project, how to build and run the Docker container, and common issues to avoid.

## Current Docker Setup

The project uses Docker for containerized deployment with the following key components:

1. **Dockerfile**: Defines the container image setup
2. **run-mcp-logic.sh/bat**: Scripts to build and run the container on Linux/Windows
3. **claude-app-config.json**: Configuration for integrating with Claude Desktop App

## Current Issues Identified

1. **Virtual Environment Path Problem**: 
   - Current Dockerfile sets `ENV PATH="/.venv/bin:$PATH"` with a leading slash, which is incorrect
   - The path should be `/app/.venv/bin` (relative to the working directory)

2. **Claude App Config Issue**:
   - Current config expects a pre-running Docker container
   - Better approach: Run directly with UV for seamless integration

## Correct Dockerfile Configuration

```dockerfile
# Dockerfile for mcp-logic with Prover9
FROM python:3.9-slim

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application
COPY . .

# Create Python virtual environment in the correct location
RUN pip install uv
RUN uv venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies with specific versions
RUN pip install --upgrade pip
RUN pip install -e .

# Install other requirements if they exist
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Create a directory to mount the local Prover9 binaries
RUN mkdir -p /usr/local/prover9-mount

# Set environment variables
ENV DOCKER_HOST=unix:///var/run/docker.sock

# Expose ports - try multiple ports in case some are in use
EXPOSE 8888 8889 8890 8891 8892

# Command to run the server
CMD ["sh", "-c", "uv --directory /app/src/mcp_logic run mcp_logic --prover-path /usr/local/prover9-mount"]
```

## Recommended Claude App Config

For direct integration without Docker:

```json
{
  "mcpServers": {
    "mcp-logic": {
      "command": "uv",
      "args": [
        "--directory", 
        "/home/ty/Repositories/mcp-logic/src/mcp_logic",
        "run", 
        "mcp_logic", 
        "--prover-path", 
        "/home/ty/Repositories/mcp-logic/ladr/bin"
      ]
    }
  }
}
```

## Building and Running with Docker

### Build the Docker Image

```bash
docker build -t mcp-logic .
```

### Run the Container

```bash
docker run -it --rm \
  -p 8888:8888 \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/ladr/bin:/usr/local/prover9-mount" \
  --name mcp-logic \
  mcp-logic
```

### Using the Run Scripts

```bash
# Linux/macOS
./run-mcp-logic.sh

# Windows
run-mcp-logic.bat
```

## Best Practices for Docker Configuration

1. **Mount Volumes for Development**: Mount source code as volumes for rapid development
2. **Use Specific Docker Tags**: Tag images with version numbers for production
3. **Keep Images Small**: Minimize image size by cleaning up after installations
4. **Proper Error Handling**: Include proper error handling in run scripts
5. **Port Management**: Check for port availability before starting the container
6. **Security**: Don't run containers as root when possible