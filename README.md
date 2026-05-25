# Code for "Mixed-Precision Jacobi Algorithms for SVD and EVD"

Supplementary code for the preprint. Implements mixed-precision preconditioned Jacobi algorithms that compute singular values and vectors, and eigenvalues and vectors, to higher accuracy than MATLAB's built-in `svd` and `eig` for ill-conditioned matrices.

## Dependencies

- **MATLAB R2025b** (or later)
- **[Advanpix Multiprecision Toolbox](https://www.advanpix.com/)** — required for `mp` (quadruple-precision arithmetic) and for computing reference solutions in `compute_error` and `get_reference`
- **OpenBLAS** and **gfortran** — only needed if rebuilding MEX files; precompiled `.mexmaca64` binaries for Apple Silicon are included

Optional (for figure export):

- `matlab2tikz`
- `cleanfigure`

## Usage

```matlab
addpath("evdalgs");
addpath("svdalgs");

% Singular value decomposition (mixed precision)
[U, S, V] = mposj(A);           % quadruple + double + single
[U, S, V] = mposj(A, 2);        % double + single only
[U, S, V] = mposj_ssd(A);       % single + double only

% Symmetric eigenvalue decomposition (mixed precision)
[V, D] = mp_pjacobi(A, "mp3");  % quadruple + double + single
[V, D] = mp_pjacobi(A, "mp2");  % double + single only
[V, D] = cjacobi(A);            % classical Jacobi (working precision only)
```

### Test scripts

| Script | Description |
|--------|-------------|
| `test1` | SVD: right singular vectors, varying condition number |
| `test2` | SVD: right singular vectors, varying column count |
| `test3` | EVD: eigenvectors, varying condition number |
| `test4` | EVD: eigenvectors, varying matrix size |
| `test5` | SVD: special test matrices (KMS, Lehmer) |
| `test6` | EVD: special test matrices (KMS, Lehmer) |
| `test7` | SVD: left singular vectors, varying condition number |

Run `mytest` to execute tests 1 through 4 in sequence.

### Rebuilding MEX files

Precompiled `.mexmaca64` binaries are provided in `svdalgs/`. To rebuild for a different MATLAB version or architecture, edit the paths in each `get_*/build_*_mex.m` to point to your local OpenBLAS and gfortran installations, then run the build script.

## Tested environment

```
OS: macOS 26.3.1 arm64
CPU: Apple M3 Pro
MATLAB: R2025b Update 1
```

## Notes

- Hardcoded paths in `matlab2tikz` calls (tests 1–7) and MEX build scripts are marked with `% EDIT THIS PATH` — adjust these to match your system.
- Tests 1–4 and 7 use `beep` and `pause()` when computed errors exceed theoretical bounds. Remove these lines for unattended execution.
