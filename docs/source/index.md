
# QuaTorch
***Quaternions in PyTorch***


```{image} /_static/logo.svg
:alt: QuaTorch Logo
:width: 30%
:target: https://pypi.org/project/quatorch/
```

**QuaTorch** is a lightweight python package providing `Quaternion`, a `torch.Tensor` subclass that represents a [Quaternion](https://en.wikipedia.org/wiki/Quaternion). It implements common special operations for quaternions such as multiplication,
conjugation, inversion, normalization, log, exp, etc. It also supports conversion to/from rotation matrix and axis-angle representation. Convenient utilities are provided together, such as spherical linear interpolation ([slerp](https://en.wikipedia.org/wiki/Slerp)) and 3D vector rotation.


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
from quatorch import Quaternion

q = Quaternion(1, 0, 0, 0)  # identity quaternion
print(q)
```

See more examples in the <project:examples.md> page and the <project:api.md>.

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


```{toctree}
:maxdepth: 2
examples.md
api
GitHub <https://github.com/egidioln/QuaTorch>
PyPI <https://pypi.org/project/quatorch/>
``` 