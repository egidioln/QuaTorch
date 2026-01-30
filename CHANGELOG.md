# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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