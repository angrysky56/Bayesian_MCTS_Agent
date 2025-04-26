# MCP-Logic Project Overview

## Purpose
MCP-Logic is an MCP (Model Context Protocol) server that provides automated reasoning capabilities using Prover9/Mace4 for AI systems. It enables logical theorem proving and logical model verification through a clean MCP interface.

## Tech Stack
- **Python 3.10+**: Core implementation language
- **MCP Server Framework**: For building the Model Context Protocol server
- **Prover9/Mace4**: The underlying automated theorem prover (LADR - Logic for Automated Deductive Reasoning)
- **Docker**: For containerization and deployment
- **Pydantic**: For data validation
- **Hatchling**: For package building

## Key Components
1. **LogicEngine Class**: Core component that interfaces with Prover9
2. **MCP Server Implementation**: Exposes logical reasoning capabilities as MCP tools
3. **Prover9/Mace4 Integration**: Automated theorem proving engine

## Available Tools
1. **prove**: Prove a logical statement using premises and a conclusion
2. **check-well-formed**: Validate that logical statements are well-formed

## Project Structure
- `/src/mcp_logic/`: Main source code
  - `server.py`: Core server implementation
  - `__main__.py`: Entry point
- `/tests/`: Test cases
- `/ladr/bin/`: Prover9/Mace4 binaries
- Docker configuration for containerized deployment
- Scripts for Windows/Linux compatibility

## Dependencies
As specified in pyproject.toml:
- Python 3.10+
- mcp >= 1.0.0
- pydantic >= 2.0.0