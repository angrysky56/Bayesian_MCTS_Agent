# System Requirements for MCP-Logic

## Core Requirements

- **Operating System**: Linux (preferred), Windows, or macOS
- **Python**: Version 3.10 or higher
- **Prover9/Mace4**: The LADR package installed and accessible

## Hardware Requirements

- No specific hardware requirements beyond what's needed to run Python and Prover9
- Minimal disk space needed for installation (~50MB)
- Minimal memory requirements for running the server

## Software Dependencies

### Python Packages
- mcp >= 1.0.0
- pydantic >= 2.0.0
- UV package manager (recommended)

### External Tools
- **Prover9/Mace4**: Automated theorem prover (installed via LADR)
- **Git**: For version control
- **Docker**: Optional, for containerized deployment

## Development Environment

### Recommended Tools
- **Visual Studio Code** with Python extensions
- **UV** for virtual environment and package management
- **Pytest** for running tests

### Optional Tools
- **Black** for code formatting
- **Flake8** for linting

## Network Requirements

- The server runs locally by default
- Port 8888 (or alternate ports 8889-8892) should be available for the server to bind to

## Permissions

- Execution permissions for the Prover9/Mace4 binaries
- Write permissions for the project directory (for logs, temporary files, etc.)

## Claude Desktop App Integration

- Claude Desktop App must be configured to start/stop the MCP-Logic server
- Configuration file (claude-app-config.json) with appropriate permissions

## Docker Requirements (if using Docker)

- Docker Engine installed and running
- Permission to build and run Docker containers
- Port mapping capabilities (for exposing the server port)
- Volume mounting capabilities (for source code and Prover9 binaries)