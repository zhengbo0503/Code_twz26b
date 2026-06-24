# Code for preprint "Computing Accurate Singular Vectors and Eigenvectors Using Mixed-Precision Jacobi Algorithms"

Test scripts and supplementary code for testing the mixed-precision preconditioned (one-sided) Jacobi algorithm for computing highly accurate singular vectors and eigenvectors. 

## Codes for Jacobi algorithms
The codes for the mixed-precision Jacobi algorithm and its one-sided variant are providede in the `svdalgs` and `evdalgs` folders, respectively. They are taken from 
 - [Code_twz26](https://github.com/zhengbo0503/Code_twz26) -- mixed-precision one-sided Jacobi algorithm for SVD, and 
 - [Code_htwz25](https://github.com/zhengbo0503/Code_htwz25) -- mixed-precision Jacobi algorithm for spectral decomposition. 

## Dependencies
Our code has been tested with the following setup:
```
OS: macOS 26.5.1 25F80 arm64 
CPU: Apple M3 Pro 
GPU: Apple M3 Pro 
Memory: 36864MiB 
MATLAB version: 25.2.0.3042426 (R2025b) Update 1
OpenBLAS version: OpenBLAS 0.3.29.dev
Advanpix version: 5.4.4 Build 16174
GNU Fortran: (Homebrew GCC 15.2.0_1) 15.2.0
```
- **[Advanpix Multiprecision Toolbox](https://www.advanpix.com/)**
- **[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.29)**
- **[GNU Fortran](https://gcc.gnu.org/wiki/GFortranBinaries)**

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

## Test scripts

| Script  | Description                                           |
| ------- | ----------------------------------------------------- |
| `test1` | SVD: right singular vectors, varying condition number |
| `test2` | SVD: right singular vectors, varying column count     |
| `test3` | EVD: eigenvectors, varying condition number           |
| `test4` | EVD: eigenvectors, varying matrix size                |
| `test5` | SVD: special test matrices (KMS, Lehmer)              |
| `test6` | EVD: special test matrices (KMS, Lehmer)              |
| `test7` | SVD: left singular vectors, varying condition number  |

## Functions 
| Function        | Description                                                              |
| --------------- | ------------------------------------------------------------------------ |
| `mposj`         | Mixed-precision one-sided Jacobi algorithm (quadruple + double + single) |
| `mposj_ssd`     | Mixed-precision one-sided Jacobi algorithm (single + double)             |
| `cjacobi`       | Two-sided Jacobi algorithm (double)                                      |
| `mp_pjacobi`    | Mixed-precision two-sided Jacobi algorithm (quadruple + double + single) |
| `eigsort`       | MATLAB `eig` with eigenvalues sorted in descending order                 |
| `compute_error` | Compute the error between computed and exact singular/eigenvectors       |

## Rebuilding MEX files

Precompiled `.mexmaca64` binaries are provided in `svdalgs/`. To rebuild for a different MATLAB version or architecture, edit the paths in each `get_*/build_*_mex.m` to point to your local OpenBLAS and gfortran installations, then run the build script.

## Notes

- Tests 1–4 and 7 use `pause()` when computed errors exceed theoretical bounds. Remove these lines for unattended execution.
