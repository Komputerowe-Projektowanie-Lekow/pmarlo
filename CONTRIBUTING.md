# Contributing to PMARLO

Thank you for your interest in contributing to PMARLO! This document provides guidelines for development and testing.

## Development Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/pmarlo.git
cd pmarlo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Project Structure

```
pmarlo/
├── src/                     # Source code
│   ├── __init__.py         # Main package exports
│   ├── pipeline.py         # Pipeline orchestration
│   ├── protein/            # Protein preparation
│   ├── replica_exchange/   # REMD functionality  
│   ├── simulation/         # MD simulation
│   ├── markov_state_model/ # MSM analysis
│   └── manager/            # Checkpoint management
├── tests/                  # Unit tests and integration tests
│   ├── data/              # Test data files (PDB, trajectories)
│   ├── conftest.py        # Pytest configuration
│   └── test_*.py          # Test modules
├── examples/              # Usage examples
├── output/                # Runtime output (git-ignored)
├── bias/                  # Runtime bias files (git-ignored)
└── docs/                  # Documentation (future)
```

### 3. Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_protein.py

# Run with coverage
pytest --cov=src

# Run tests in parallel
pytest -n auto
```

### 4. Code Quality

```bash
# Format code
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/

# Type checking
mypy src/
```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting (88 character line limit)
- Use type hints where possible
- Write descriptive docstrings

### Testing

- Write unit tests for all new functionality
- Use pytest fixtures for common test data
- Test both success and failure cases
- Integration tests should be marked with appropriate skip conditions

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions and classes
- Include usage examples for new features
- Update type hints and parameter documentation

## File Organization Rules

### What Goes Where

#### `src/` - Source Code
- **Package code only** - no test data, no examples
- Each module should have a clear, single responsibility
- Use proper imports and type hints

#### `tests/` - Testing
- **Unit tests and integration tests**
- `tests/data/` - Test input files (PDB files, trajectories)
- `conftest.py` - Shared pytest fixtures
- Test files should mirror the source structure

#### `examples/` - Usage Examples
- **Complete, runnable examples**
- Demonstrate different usage patterns
- Should handle missing dependencies gracefully

#### `output/` and `bias/` - Runtime Data
- **Git-ignored** - these are generated during runs
- Users can clean these safely
- Should not contain any source code or examples

### Test Data Guidelines

- Keep test files small (< 10MB each)
- Use realistic but minimal test cases
- Document what each test file represents
- Place in `tests/data/` directory

## Making Changes

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Development Workflow

1. Write tests first (TDD recommended)
2. Implement the feature
3. Run tests and ensure they pass
4. Update documentation
5. Run code quality checks

### 3. Testing Your Changes

```bash
# Quick test
pytest tests/test_your_module.py

# Full test suite
pytest

# Test examples work
python examples/basic_usage.py
python examples/advanced_usage.py

# Test package installation
pip install -e .
python -c "from pmarlo import Pipeline; print('✅ Import works')"
```

### 4. Commit and Push

```bash
git add .
git commit -m "feat: add new feature"
git push origin feature/your-feature-name
```

## Common Development Tasks

### Adding a New Class

1. Create the class in appropriate module
2. Add to `__init__.py` exports if public
3. Write comprehensive unit tests
4. Add usage example
5. Update type hints and documentation

### Adding New Dependencies

1. Add to `requirements.txt` (runtime dependencies)
2. Add to `requirements-dev.txt` (development dependencies)  
3. Update `pyproject.toml` dependencies
4. Test that installation still works

### Handling Computational Dependencies

Some dependencies (OpenMM, RDKit) are computationally expensive:

- Make imports conditional where possible
- Use pytest.skip for tests requiring heavy dependencies
- Provide graceful fallbacks in examples
- Document system requirements clearly

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test package
5. Create release on GitHub
6. Upload to PyPI (maintainers only)

## Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones
- Be descriptive in issue titles and descriptions

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn the codebase
- Focus on what's best for the project

Thank you for contributing to PMARLO! 🧬