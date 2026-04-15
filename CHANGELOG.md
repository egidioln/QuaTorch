# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.3.0] - 2026-04-16

### Feature
- `from_axis_angle()` now supports broadcasting.
- Improved handling of edge cases(~180-degree rotations) in `from_rotation_matrix()`.

### Fixed
- Bug where identity matrix was not handled correctly in `from_rotation_matrix()`, reported in https://github.com/egidioln/QuaTorch/issues/7

## [0.2.1] - 2026-03-07

### Fixed
- Bug where matrices with negative trace would be treated as symmetric in `from_rotation_matrix()`, reported in https://github.com/egidioln/QuaTorch/issues/5


## [0.2.0] - 2026-01-30

### Feature
- Added support for `torch.compile` without graph breaks for `mul`, `rotate_vector`, and `slerp` methods.


## [0.1.3] - 2025-12-03

### Fixed
- Handle symmetric matrices in `from_rotation_matrix()`.

## [0.1.2] - 2025-10-31

### Feature
- `.conj()` is now an alias for `.conjugate()`

### Fixed
- `to()` returns a `Quaternion` (instead of a `torch.Tensor`) 

## [0.1.1] - 2025-10-01

### Fixed

- Some operations were not returning correctly quaternion types
- Fixed `imag`and `real`properties, which were treated was functions
