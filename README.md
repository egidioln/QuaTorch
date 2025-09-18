
<img src="https://raw.githubusercontent.com/egidioln/QuaTorch/refs/heads/main/docs/source/_static/logo.svg" width="400px">

***Quaternions in PyTorch***
# QuaTorch
<!--[![Release](https://img.shields.io/github/v/release/actions/deploy-pages?label=Release&logo=github)](https://github.com/actions/deploy-pages/releases/latest) -->
[![cov](https://raw.githubusercontent.com/egidioln/QuaTorch/refs/heads/gh-pages/badges/coverage.svg)](https://github.com/egidioln/quatorch/actions) 
[![Tests](https://img.shields.io/github/actions/workflow/status/egidioln/QuaTorch/pytest.yml?label=Tests&logo=github)](https://github.com/egidioln/QuaTorch/actions/workflows/pytest.yml) 
[![Docs](https://img.shields.io/github/actions/workflow/status/egidioln/QuaTorch/docs.yml?label=Docs&logo=github)](https://egidioln.github.io/QuaTorch/) 

The package `quatorch` provides `Quaternion`, a `torch.Tensor` subclass that represents a [Quaternion](https://en.wikipedia.org/wiki/Quaternion). It implements common operations in following quaternion algebra such as multiplication,
conjugation, inversion, normalization, log, exp, etc. It also supports conversion to/from rotation matrix and axis-angle representation. Convenient utilities are provided together, such as spherical linear interpolation ([slerp](https://en.wikipedia.org/wiki/Slerp)) and 3D vector rotation.

## Highlights

- Quaternion type: `quatorch.Quaternion` (subclass of `torch.Tensor`).
- Element-wise and algebraic ops implemented: `+`, `-`, `*` (quaternion product and scalar mul),
	`abs` (norm), `conjugate`, `inverse`, `normalize`, `to_rotation_matrix`, and more.
- Utilities: `from_rotation_matrix`, `from_axis_angle`, `to_axis_angle`, `rotate_vector`, `slerp`,
	`log`, `exp`, and `pow`.

## Installation

This project targets Python 3.10+ and requires PyTorch. Install via pip (recommended):

```bash
pip install quatorch
```

Or install editable/development mode:

```bash
git clone 
cd quatorch
pip install -e .
```

See `pyproject.toml` for dependency details (dev deps include `pytest`).

## Quick start

Basic usage examples using PyTorch tensors and `Quaternion`:

```python
import torch
from quatorch.quaternion import Quaternion

# Create a quaternion from four scalars (W, X, Y, Z)
q = Quaternion(1.0, 0.0, 0.0, 0.0)

# Or from a tensor of shape (..., 4)
q2 = Quaternion(torch.tensor([0.9239, 0.3827, 0.0, 0.0]))  # 45° around X

# Normalize
q2 = q2.normalize()

# Quaternion multiplication (rotation composition)
q3 = q * q2

# Rotate a vector
v = torch.tensor([1.0, 0.0, 0.0])
v_rot = q2.rotate_vector(v)

# Convert to rotation matrix
R = q2.to_rotation_matrix()

# Slerp between quaternions
t = 0.5
q_mid = q.slerp(q2, t)
```

## API notes

- Construction:
	- `Quaternion(data: torch.Tensor)` where `data.shape[-1] == 4`.
	- `Quaternion(w, x, y, z)` accepts scalars or tensors broadcastable to the same shape.

- Shape requirements:
	- The last dimension must be size 4 for quaternion tensors (W, X, Y, Z).

- Interoperability:
	- The class implements several `torch.*` functions via a small dispatcher so
		many PyTorch APIs behave sensibly with `Quaternion` objects.

See the source in `src/quatorch/quaternion.py` for the full implementation and more
helper methods.

## Running tests

This repository includes unit tests using `pytest` under `test/unit_tests`.

From the project root, run:

```bash
uv run --with=. pytest 
```

## Contributing

Contributions are welcome. A few ideas:

- Add more conversions and higher-level utilities (e.g., batch rotation helpers).
- Improve numeric stability and add property-based tests.
- Add docs and usage notebooks / examples.

Please open issues or pull requests on the repository.

## License

MIT — see the [LICENSE.md](./LICENSE.md) for author/maintainer information.

## Contact

Maintainer: Lucas N. Egidio <lucasegidio1@gmail.com>

