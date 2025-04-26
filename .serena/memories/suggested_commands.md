# Suggested Commands for MCP-Logic Development

## Environment Setup

```bash
# Clone the repository (if not already done)
git clone https://github.com/angrysky56/mcp-logic
cd mcp-logic

# Create and activate Python virtual environment using UV
pip install uv
uv venv
source .venv/bin/activate

# Install the package in development mode
uv pip install -e .
```

## Running the Server Locally

```bash
# Run directly with UV (non-Docker)
uv --directory src/mcp_logic run mcp_logic --prover-path /home/ty/Repositories/mcp-logic/ladr/bin
```

## Docker Commands

```bash
# Build the Docker image
docker build -t mcp-logic .

# Run the Docker container
docker run -it --rm \
  -p 8888:8888 \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/ladr/bin:/usr/local/prover9-mount" \
  --name mcp-logic \
  mcp-logic

# Run with the provided script
./run-mcp-logic.sh
```

## Testing

```bash
# Run tests
uv pip install pytest
pytest
```

## Development Utilities

```bash
# Check for Python syntax errors
find src -name "*.py" -exec python -m py_compile {} \;

# List files
find src -type f | sort

# Check server logs
tail -f /tmp/mcp_logic*.log  # If logs are saved to file

# Format code (if you install black)
uv pip install black
black src/
```

## Integration with Claude Desktop App

To configure MCP-Logic to run with Claude Desktop, use the claude-app-config.json.
For direct integration (recommended), update the config to:

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

## Prover9 Commands

```bash
# Test Prover9 directly
/home/ty/Repositories/mcp-logic/ladr/bin/prover9 -f input_file.in

# Check if Prover9 is working
/home/ty/Repositories/mcp-logic/ladr/bin/prover9 --help
```
