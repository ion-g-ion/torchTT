# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- API documentation for `torchtt.functional`, explaining its use by `torchtt.nn.TTDensityLayer` and linking the existing tutorials (#49).
- `torchtt.matvec`, `torchtt.matmat` and `torchtt.hadamard` compute products with rank truncation (e.g. `(A @ x).round(eps, rmax)`) and choose the algorithm with the `method` argument: `'direct'`, `'dmrg'`, `'amen'` or `'swap'`. The same functions are available as `TT.matvec`, `TT.matmat` and `TT.hadamard` (#48).
- `torchtt.methods` with the frozen dataclasses `Direct`, `DMRG`, `AMEn` and `Swap`. Passing an instance instead of a name changes the parameters of a method, e.g. `method=torchtt.methods.DMRG(nswp=40)`.
- Documentation on which method to choose for a product (#47).
- Documentation of `torchtt.grad`: its relation to PyTorch autograd, which operations can be differentiated and the side effects of `grad()` (#47).
- The DMRG elementwise product also accepts TT matrices.

### Changed
- Docs: the overview section "About the package" is renamed "Operations in the TT format" and no longer lists the examples. Instead, every example page links to Google Colab and to the notebook on GitHub (`nbsphinx_prolog` in `conf.py`).

### Fixed
- `GaussianBasis.interpolating_points()` now returns rows for sample points and columns for basis functions, matching `BSplineBasis` and correcting interpolation with nonuniform Gaussian centers.
- Removed the unused, unfinished `torchtt.functional.pdf` module, which contained syntax errors (#50). The basis classes and `torchtt.nn.TTDensityLayer` remain available.
- `from torchtt import *` failed because `__all__` listed `cpp_available` and `fast_hadamard`, which do not exist (the functions are `cpp_enabled` and `fast_hadammard`).
- The Python DMRG products changed the cores of the initial guess passed to them.
- AMEn products returned wrong results for complex tensors; they now raise `IncompatibleTypes`.
- The rank of AMEn products could exceed `rmax` because of the rank enrichment.

### Deprecated
- `TT.fast_matvec`, `torchtt.fast_mv`, `torchtt.fast_mm`, `torchtt.fast_hadammard`, `torchtt.amen_mv`, `torchtt.amen_mm` and `torchtt.dmrg_hadamard`. They still work and emit a `DeprecationWarning` that names the replacement.

## [0.5.0] - 2026-08-21

### Added
- Composable `Transform` API for `TTDensityLayer` in `torchtt.nn`: `AffineTransform`, `Rank1Shear`, `TriangularShear`, `TriangularPolyShear`, `SinhArcsinhWarp`, and `ComposedTransform` for chaining them.
- `BSplineBasis.eval_sparse()`: returns only the `deg + 1` values that are nonzero at each point together with their basis indices, so work and memory no longer grow with the number of basis functions.
- Gram matrices for the spline basis, including derivative pairings.
- Fokker-Planck PINN example (`examples/fokker_planck_pinn.py` / `.ipynb`).
- Tests for the TT density layer (`tests/test_tt_density_layer.py`), plus substantially wider basis and cross-approximation test coverage.

### Fixed
- Cross approximation (`dmrg_cross` and `function_interpolate`) discarded the `R` factor of the rank-enriching QR whenever the enriched block was *not* rank-deficient, silently returning a wrong interpolant. `R` is now always applied.
- The added-rank count in the left-to-right sweep of `dmrg_cross` was derived from the QR's `Q` rather than its `R`, padding `V` to the wrong width when the enriched block had fewer rows than columns.
- Further GPU fixes in cross approximation: `_maxvol` built its index tensor on the CPU, and several intermediates were cast without also being moved to `device`.
- `BSplineBasis` now invalidates its cached span bounds on `.to()`, `.cuda()`, `.float()` and friends, since a cast can collapse neighbouring knots.

### Changed
- **Breaking**: `TTDensityLayer(..., linear_transformation=...)` is replaced by `TTDensityLayer(..., transform=...)`, which takes a `Transform` instance.
- **Breaking**: the static `TTDensityLayer.input_requireemnt(N, R, linear_transformation)` becomes the instance method `TTDensityLayer.input_requirement()` (typo fixed).
- `examples/deep_tt_density/` (`generate_data.py`, `main.py`, `trainer.py`) is consolidated into a single `examples/deep_tt_density.py` / `.ipynb`.
- `_build_two_core_eval_index` moved from `torchtt.interpolate` to `torchtt._dmrg` and rewritten without `kron`.
- Documentation version now tracks the package version instead of being pinned to `2.0`.

## [0.4.0] - 2026-06-28

### Added
- Deep TT density estimation and deep TT pdf estimation
- AMEN cross and approximation, plus a new layer
- Compressed layer functionality
- Bayesian inference example and other new examples

### Fixed
- Cross approximation on GPU (resolved unravel_index issue)
- Various testing fixes and general file cleanup

### Changed
- Updated JOSS paper (`paper.md` and `paper.bib`)
- Updated documentation and GitHub workflows
- Renamed examples
