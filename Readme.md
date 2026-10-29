# SpMM benchmark

This CPU benchmark compares custom OpenMP CSR, Intel MKL, and Compressed Sparse Blocks (CSB) implementations of `C = A * B`, using double-precision values. For an `m × n` sparse matrix `A` and an `n × k` dense matrix `B`, the output `C` has `m × k` entries. `RHSDIM` is `k`, the number of columns of `B`.

## Requirements

Build and run on Linux (including WSL) with GNU Make, a C++20-capable GNU compiler (`g++`) with OpenMP, and Intel oneMKL headers and libraries. Initialize your oneAPI environment before building, for example:

```bash
source /opt/intel/oneapi/setvars.sh
```

If oneMKL is installed elsewhere, set `MKLROOT` to its installation directory. `Makefile.gnu2` expects headers under `$MKLROOT/include` and libraries under `$MKLROOT/lib/intel64`; its fallback is `/opt/intel/oneapi/mkl/latest`. The MKL shared libraries must also be on the runtime library search path (normally configured by the oneAPI environment). Intel Advisor is not required for this build.

## Build with Makefile.gnu2

Use **`spmm/Makefile.gnu2`** for the benchmark. Always specify `-f Makefile.gnu2`: this repository has no default `Makefile`. This configuration uses GNU OpenMP (`-fopenmp`) and MKL's GNU threading library (`mkl_gnu_thread`). The other makefiles (`Makefile.gnu`, `Makefile.intel`, and `Makefile.original`) are older or alternative configurations; they are not the recommended build path.

### Set the number of columns of B

Edit this definition near the top of `spmm/spmm.cpp`, before the CSB include:

```cpp
#define RHSDIM 4   // number of columns in B, used for CSB
```

Set it to the desired positive column count (for example, `1`, `4`, `16`, `32`, or `64`), save the file, and rebuild. The checked-in source currently defines `RHSDIM` as `64`. CSB uses this compile-time constant for its packed row arrays and template instantiations. Changing only the runtime column argument does not change the compiled CSB width. **The runtime column count must equal the value of `RHSDIM` used to compile that executable.** The program currently does not enforce this equality; a mismatch can produce incorrect results or out-of-bounds accesses.

From the repository root, after setting `RHSDIM` to `4`:

```bash
cd spmm
make -f Makefile.gnu2 -B
./spmm /path/to/matrix.mtx 4 16
```

The default output is `spmm`. `-B` forces recompilation, which is useful because executable files are already tracked in this repository and the makefile only lists `spmm.cpp` as a dependency. Rebuild after changing CSB headers as well.

Useful build commands (run from `spmm/`):

```bash
# Build or rebuild the default executable.
make -f Makefile.gnu2 -B

# Build a named executable; first set RHSDIM to 4 in spmm.cpp.
make -f Makefile.gnu2 -B TARGET=spmm_RHSDIM4

# Override compiler or MKL location if needed.
make -f Makefile.gnu2 -B CXX=g++ MKLROOT=/path/to/mkl

# Remove the default executable and *.o files.
make -f Makefile.gnu2 clean

# Remove a particular named executable and *.o files.
make -f Makefile.gnu2 clean TARGET=spmm_RHSDIM4
```

`TARGET` controls only the output filename. It does not set `RHSDIM`. Likewise, passing `RHSDIM=4` to make does not update the source definition with the current makefile. Set the definition in `spmm.cpp` explicitly before each build. The build uses `-march=native`, so rebuild on the intended machine or a compatible CPU.

### What are spmm_RHSDIM1, spmm_RHSDIM4, etc.?

These are saved compiled executables of the same benchmark, named by the intended compile-time width of `B`:

| Executable | Intended compiled `RHSDIM` | Required runtime column argument |
|---|---:|---:|
| `spmm_RHSDIM1` | 1 (SpMV: one right-hand side) | `1` |
| `spmm_RHSDIM4` | 4 | `4` |
| `spmm_RHSDIM16` | 16 | `16` |
| `spmm_RHSDIM64` | 64 | `64` |

The filename is a convention, not a check of the binary's contents. Rebuild each binary from source with its matching definition before benchmarking. For example, to retain both a one-column and a four-column build:

1. Set `#define RHSDIM 1` in `spmm.cpp`, then run `make -f Makefile.gnu2 -B TARGET=spmm_RHSDIM1`.
2. Set `#define RHSDIM 4` in `spmm.cpp`, then run `make -f Makefile.gnu2 -B TARGET=spmm_RHSDIM4`.
3. Run the matching executable and column argument:

```bash
./spmm_RHSDIM1 /path/to/matrix.mtx 1 16
./spmm_RHSDIM4 /path/to/matrix.mtx 4 16
```

Editing `RHSDIM` later does not change executables that were already built. `spmm` is simply the default output name and has whichever width was used when it was last compiled.

## Run the benchmark

```bash
./spmm <matrix_file.mtx> <n_columns_B> <num_threads>
```

All three arguments are required. `n_columns_B` must match the executable's compiled `RHSDIM`, and `num_threads` must be positive. The program sets both OpenMP and MKL thread counts from `num_threads`. It generates a row-major dense `B` using `B[i,j] = sin(i+j)`, then prints `spmm_csr time`, `spmm_mkl time`, and `spmm_csb time` in seconds. Matrix loading and format conversion are outside those kernel timings; CSB also performs an untimed warm-up. The separate `[OMP sanity]` timing is not an SpMM result.

Input matrices must be Matrix Market coordinate files with real, integer, or pattern entries. The loader expands symmetric entries, sorts coordinates, and sums duplicates. Download and extract the desired `.mtx` files from the dataset links below; they are not bundled with the repository.

The source includes a C++ comparison helper, but the three comparison print statements near the end of `main` are commented out by default. Uncomment them and rebuild if you want pairwise result comparisons.

## Dataset
The following datasets are used to evaluate performance across the implementations:

Blocking:

| Name                  | Rows        | Cols        | Nonzeros      | Kind                                           | Link                                                                 |
|-----------------------|------------:|------------:|--------------:|-----------------------------------------------|----------------------------------------------------------------------|
| road_usa              | 23,947,347 | 23,947,347 | 57,708,624    | Undirected Graph                               | [link](https://sparse.tamu.edu/DIMACS10/road_usa)                   |
| hugebubbles-00010     | 19,458,087 | 19,458,087 | 58,359,528    | Undirected Graph                               | [link](https://sparse.tamu.edu/DIMACS10/hugebubbles-00010)          |
| asia_osm              | 11,950,757 | 11,950,757 | 25,423,206    | Undirected Graph                               | [link](https://sparse.tamu.edu/DIMACS10/asia_osm)                   |
| 333SP                 | 3,712,815  | 3,712,815  | 22,217,266    | Undirected Graph                               | [link](https://sparse.tamu.edu/DIMACS10/333SP)                      |
| Spielman_k300         | 9,045,202  | 9,045,202  | 27,226,204    | Undirected Weighted Graph                      | [link](https://sparse.tamu.edu/FlowIPM22/Spielman_k300)             |

Scale-free:

| Name                  | Rows        | Cols        | Nonzeros      | Kind                                           | Link                                                                 |
|-----------------------|------------:|------------:|--------------:|-----------------------------------------------|----------------------------------------------------------------------|
| com-Orkut             | 3,072,441  | 3,072,441  | 234,370,166   | Undirected Graph With Communities (Social Network) | [link](https://sparse.tamu.edu/SNAP/com-Orkut)                   |
| twitter7              | 41,652,230 | 41,652,230 | 1,468,365,182 | Directed Graph (Social Network)                | [link](https://sparse.tamu.edu/SNAP/twitter7)                        |
| com-LiveJournal       | 3,997,962  | 3,997,962  | 69,362,378    | Undirected Graph With Communities (Social Network) | [link](https://sparse.tamu.edu/SNAP/com-LiveJournal)           |
| GAP-web               | 50,636,151 | 50,636,151 | 1,930,292,948 | Directed Weighted Graph (Web)                  | [link](https://sparse.tamu.edu/GAP/GAP-web)                         |
| uk-2002               | 18,520,486 | 18,520,486 | 298,113,762   | Directed Graph (Web)                           | [link](https://sparse.tamu.edu/LAW/uk-2002)                         |
| indochina-2004        | 7,414,866  | 7,414,866  | 194,109,311   | Directed Graph (Web)                           | [link](https://sparse.tamu.edu/LAW/indochina-2004)                  |

Diagonal:

| Name                  | Rows        | Cols        | Nonzeros      | Kind                                           | Link                                                                 |
|-----------------------|------------:|------------:|--------------:|-----------------------------------------------|----------------------------------------------------------------------|
| stokes                | 11,449,533 | 11,449,533 | 349,321,980   | Semiconductor Process Problem                  | [link](https://sparse.tamu.edu/VLSI/stokes)                          |
| Queen_4147            | 4,147,110  | 4,147,110  | 316,548,962   | 2D/3D Problem                                  | [link](https://sparse.tamu.edu/Janna/Queen_4147)                     |
| rajat31               | 4,690,002  | 4,690,002  | 20,316,253    | Circuit Simulation Problem                      | [link](https://sparse.tamu.edu/Rajat/rajat31)                        |
