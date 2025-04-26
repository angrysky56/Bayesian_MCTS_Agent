# Task Completion Checklist for MCP-Logic

When completing a task on the MCP-Logic project, follow this checklist to ensure that your changes are properly implemented and ready for integration.

## Code Quality Checks

- [ ] Ensure type hints are used consistently in new code
- [ ] Add appropriate docstrings to new functions, methods, and classes
- [ ] Follow the project's naming conventions
- [ ] Implement proper error handling with informative error messages
- [ ] Include logging statements at appropriate levels (debug, info, warn, error)

## Testing

- [ ] Run existing tests to make sure your changes don't break anything
  ```bash
  pytest
  ```
- [ ] Add new tests for your changes when appropriate
- [ ] Verify that the server starts correctly with your changes
  ```bash
  uv --directory src/mcp_logic run mcp_logic --prover-path /home/ty/Repositories/mcp-logic/ladr/bin
  ```

## Integration Testing

- [ ] If your changes involve the Prover9 integration, test directly with Prover9
  ```bash
  /home/ty/Repositories/mcp-logic/ladr/bin/prover9 -f test_input.in
  ```
- [ ] Test the MCP server through the Claude Desktop App if relevant
- [ ] Verify that your changes work correctly in Docker if they involve Docker-specific functionality

## Documentation

- [ ] Update README.md if your changes introduce new features or modify existing ones
- [ ] Document any new configuration options or dependencies
- [ ] Update any example code that might be affected by your changes

## Final Verification

- [ ] Review your changes one more time to ensure they meet requirements
- [ ] Check that all imports are necessary and correctly organized
- [ ] Verify that virtual environment handling is correct if you modified setup files
- [ ] Make sure Docker configuration is consistent with your changes if relevant

## Deployment (if applicable)

- [ ] Rebuild Docker image if changes affect Docker setup
  ```bash
  docker build -t mcp-logic .
  ```
- [ ] Update claude-app-config.json if necessary for integration with Claude Desktop App
- [ ] Test the complete deployment flow one more time