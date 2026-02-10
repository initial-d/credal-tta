# Contributing to Credal-TTA

Thank you for your interest in contributing to Credal-TTA! This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone repository
git clone https://github.com/anonymous-repo/credal-tta.git
cd credal-tta

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/
```

## Code Style

We follow PEP 8 style guidelines. Before submitting, please run:

```bash
# Format code
black credal_tta/ experiments/ tests/

# Check style
flake8 credal_tta/ experiments/ tests/
```

## Adding New Features

### Adding a New TSFM Wrapper

1. Create a new class in `credal_tta/models/wrappers.py`:

```python
class NewModelWrapper(TSFMWrapper):
    def __init__(self, **kwargs):
        # Initialize your model
        pass
    
    def predict(self, context: np.ndarray, prediction_length: int = 1) -> float:
        # Implement prediction logic
        pass
```

2. Add tests in `tests/test_credal_tta.py`

### Adding a New Baseline Method

1. Add function in appropriate experiment script:

```python
def new_baseline(time_series, model, **params):
    """Your baseline implementation"""
    predictions = []
    # ... implementation
    return np.array(predictions)
```

2. Add to comparison in experiment scripts

## Testing

All new features should include tests. We use pytest:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_credal_tta.py::TestCredalSet

# Run with coverage
pytest --cov=credal_tta tests/
```

## Documentation

- Update docstrings for all public functions
- Add examples to `examples/` directory
- Update `docs/API.md` for new API features

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community

## Questions?

Open an issue or contact the maintainers at sa613403@mail.ustc.edu.cn
