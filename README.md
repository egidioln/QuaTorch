
<img src="https://raw.githubusercontent.com/egidioln/QuaTorch/refs/heads/main/docs/source/_static/logo.svg" width="400px">

***Quaternions in PyTorch***
# QuaTorch
[![Release](https://img.shields.io/github/v/release/egidioln/QuaTorch?label=Release&logo=github)](https://github.com/egidioln/QuaTorch/releases/latest)
[![cov](https://raw.githubusercontent.com/egidioln/QuaTorch/refs/heads/gh-pages/badges/coverage.svg)](https://github.com/egidioln/quatorch/actions) 
[![Tests](https://img.shields.io/github/actions/workflow/status/egidioln/QuaTorch/pytest.yml?label=Tests&logo=github)](https://github.com/egidioln/QuaTorch/actions/workflows/pytest.yml) 
[![Docs](https://img.shields.io/github/actions/workflow/status/egidioln/QuaTorch/docs.yml?label=Docs&logo=github)](https://egidioln.github.io/QuaTorch/) 

**QuaTorch** is a lightweight python package providing `Quaternion`, a `torch.Tensor` subclass that represents a [Quaternion](https://en.wikipedia.org/wiki/Quaternion). It implements common special operations for quaternions such as multiplication,
conjugation, inversion, normalization, log, exp, etc. It also supports conversion to/from rotation matrix and axis-angle representation. Convenient utilities are provided together, such as spherical linear interpolation ([slerp](https://en.wikipedia.org/wiki/Slerp)) and 3D vector rotation.

## Highlights

- Quaternion type: `quatorch.Quaternion` (subclass of `torch.Tensor`).
- Element-wise and algebraic ops implemented: `+`, `-`, `*` (quaternion product and scalar mul),
	`abs` (norm), `conjugate`, `inverse`, `normalize`, `to_rotation_matrix`, and more.
- Utilities: `from_rotation_matrix`, `from_axis_angle`, `to_axis_angle`, `rotate_vector`, `slerp`,
	`log`, `exp`, and `pow`.
- Compatible with `torch.compile` without graph breaks.

## Installation

This project targets Python 3.10+ and requires PyTorch. Install via pip (recommended):

```bash
pip install quatorch
```

Or install editable/development mode:

```bash
git clone 
cd QuaTorch
pip install -e .
```

## Quick start

Basic usage examples using PyTorch tensors and `Quaternion`:

```python
import torch
from quatorch import Quaternion

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

- Order definition:
	- The quaternion $q=w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$ is represented by an ordered tuple $(w, x, y, z)$ and this is the expected order for a quaternion in the whole library (i.e., watch out for XYZW-ordered incoming data).


- Construction:
	- `Quaternion(data: torch.Tensor)` where `data` has `.shape[-1] == 4`. An arbitrary leading shape is supported in all operations.
	- `Quaternion(w, x, y, z)` accepts scalars or tensors broadcastable to the same shape.


- Interoperability:
	- The class implements several `torch.*` functions via a small dispatcher so
		many PyTorch APIs behave sensibly with `Quaternion` objects.


## Running tests

This repository includes unit tests using `pytest` under `test/unit_tests`.

From the project root, run:

```bash
uv run --with=.[cu128] pytest 
```

or 

```bash
uv run --with=.[cpu] pytest 
```

## Contributing

Contributions are welcome! In particular:
- Bug reports and feature requests
- Optimizing performance
- Helping improving documentation
