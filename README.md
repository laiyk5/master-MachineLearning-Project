# Machine Learning Course Project

## Team Information
- **Team Member 1:** Yikai LAI (赖奕恺)
- **Team Member 2:** Zhang Yao

## Project Selection
- **Project:** Default Project (with Bonus Points target)

## Course Information
- Course: Machine Learning (PA-3)
- Institution: City University of Hong Kong (CityUHK)

## Project Structure

```
MachineLearning/
├── data/               # Dataset and data processing scripts
│   ├── raw/           # Original data
│   └── processed/     # Cleaned/processed data
├── models/            # Saved model files
├── notebooks/         # Jupyter notebooks for exploration
│   └── 01_exploration.ipynb
├── src/               # Source code
│   ├── __init__.py
│   ├── preprocessing.py   # Data cleaning utilities
│   ├── features.py        # Feature engineering
│   └── utils/             # Helper functions
├── experiments/       # Experiment logs and configs
├── results/           # Results, figures, and outputs
│   ├── figures/
│   └── metrics/
├── docs/              # Documentation
│   └── PA-3-courseproject.pdf  # Project requirements
├── tests/             # Unit tests
├── .github/           # GitHub Actions CI/CD
├── .vscode/           # VS Code settings
├── pyproject.toml     # Project dependencies and config
├── .gitignore         # Git ignore rules
├── LICENSE            # MIT License
└── README.md          # This file
```

## Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for fast dependency management

### Setup

```bash
# Navigate to project directory
cd master/MachineLearning

# Create virtual environment and install dependencies
uv sync

# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run linting
uv run ruff check src tests
```

### Development Workflow

```bash
# Run Jupyter notebook
uv run jupyter notebook notebooks/

# Format code
uv run black src tests

# Type checking
uv run mypy src
```

## Goals

1. ✅ Complete the default project requirements
2. 🎯 Achieve bonus points through extended analysis/implementations
3. 📝 Maintain clean, documented, and tested code

## Timeline

- [x] Project setup and environment configuration
- [ ] Data exploration and preprocessing
- [ ] Model implementation
- [ ] Experiments and tuning
- [ ] Results analysis and bonus work
- [ ] Final report and submission

## Notes

- Project requirements: See `docs/PA-3-courseproject.pdf`
- Daily progress tracked in `memory/` folder
- Use `uv` for all Python operations (faster than pip)
