# Changelog

All notable changes to this project will be documented in this file.

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
