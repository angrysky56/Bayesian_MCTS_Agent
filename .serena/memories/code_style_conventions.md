# Code Style and Conventions for MCP-Logic

## Python Style

The project follows standard Python conventions with the following specifics:

### Type Hints
- Type hints are used throughout the codebase
- The project uses modern Python typing features (e.g., `dict[str, Any]` rather than `Dict[str, Any]`)
- Union types are expressed with the pipe operator (`|`) rather than `Union`
- Optional parameters are marked with the `|` or `Optional` type annotation

### Docstrings
- Functions and classes use triple-quoted docstrings
- Simple, descriptive docstrings that explain purpose and parameters
- No specific docstring format (like NumPy or Google style) is rigidly followed

### Naming Conventions
- `snake_case` for variables, functions, methods, and modules
- `PascalCase` for classes
- Private methods and functions are prefixed with underscore (e.g., `_create_input_file`)

### Code Organization
- Logical grouping of related functions
- Clear separation of concerns between the logic engine and MCP server components

## Project Specific Conventions

### Error Handling
- Extensive error handling with detailed error messages
- Error information is provided in structured dictionaries
- Logging is used for debugging and tracing application flow

### Logging
- Uses Python's standard logging module
- Debug-level logging for detailed operation tracking
- Informational logging for major application state changes
- Error logging for exceptions and problems

### MCP Server Design
- Tools are defined with descriptive names and clear input schemas
- JSON Schema is used for input validation
- Responses are structured and typed properly for MCP protocol

## Development Workflow

### Testing
- Tests should be written for new functionality
- Pytest is the testing framework of choice
- Tests are located in the `/tests` directory

### Version Control
- Commit messages should be descriptive and explain the purpose of changes
- Feature branches should be used for new development
- PRs should be kept small and focused on specific changes

## Dependencies Management
- Direct dependencies are listed in pyproject.toml
- Minimal dependencies are preferred
- Specific version constraints are used to ensure compatibility