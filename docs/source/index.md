
# QuaTorch

QuaTorch is a small, dependency-light library for working with quaternions in Python. It provides a compact quaternion type, arithmetic, conversions, and utilities that are convenient for numerical work and testing.

## Quick links

- Source: https://github.com/egidioln/QuaTorch

## Installation

Install from PyPI (recommended when published) or install in editable/development mode from source:

```bash
pip install quatorch
```
or
```bash
pip install -e .
```


## Quick start

Create and use quaternions:

```py
from quatorch.quaternion import Quaternion

q = Quaternion(1, 0, 0, 0)  # identity quaternion
print(q)
```

## Run tests

This project uses pytest. From the repository root run:

```bash
uv run --with=. pytest
```

## Development notes

- Linting and formatting: Ruff is configured for formatting on save in the workspace.
- Tests are discovered under `test` and pytest configuration is in `pyproject.toml`.

## Contributing

Contributions are welcome. Please open issues or pull requests on GitHub. Follow the existing code style and add tests for new behavior.

## License

This project is MIT licensed — see the `LICENSE.md` file for details.

