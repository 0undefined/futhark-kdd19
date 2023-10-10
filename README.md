# futhark-kdd19

This repository has updated implementations of the bfast algorithm, updated to
work with newer futhark compiler versions.

Forked from [diku-dk](https://github.com/diku-dk/futhark-kdd19)

## Bfast Performance Evaluation

Folder `bfast-c` contains a OpenMP-parallelized CPU version of bfast; the code was carefully written to be in a good form for vectorization, but it is ultimately up to gcc to extract it, i.e., no vectorization intrinsics have been used. The datasets are generated internally.

Folder `bfast-futhark` contain various bfast versions of the code written in Futhark. This includes individual computational kernels of bfast including batched-matrix-matrix multiplication and batched matrix inversion. Also datasets and means of generating datasets are provided. Details are as follows:

## (I) Futhark Code for BFast

### Datasets:

The datasets are located in `bfast-futhark/data`: dataset `sahara.in` and `peru.in` correspond to actual images taken from Sahara and Peru. There are a bunch of contrived datasets. The contrived datasets have been generated with the Futhark program `bfast-futhark/data/gen-datasets/gen-data.fut`, which receive as parameters in order: the number of pixels in the image (`M`), the length of the data series (`N`), the number of pixels used for training (`n`), and the fraction of `NaN` values (`nanfreq`). For example `nanfreq = 0.5` corresponds to half the values being `NaN`.

For example `d-32768-256-128.in.gz` corresponds to a randomly generated image in which the image has `32768` pixels (`M`), the time-series length is `256` (`N`) and half of the time series (`n = 128`) is used for training. In all contrived datasets `NaN` values appear with a frequency of `50%`. The Peru dataset has `M = 111556`, `N = 235`, and `n = 113`, and the `nanfreq` is around `0.75` (`75%` of values are `NaN`).

An estimator of the number of floating point operations (Flops) for a dataset is implemented in Futhark program `perf-calculator.fut`. This estimates the Flops as they appear in the `bfast.fut` program. Floating point operations include, addition, multiplication, division, casts, `isnan`, special functions (`sqrt`, `log`), etc. The input of `perf-calculator.fut` is any dataset of Bfast.

Finally, in folder `bfast-futhark/data`, the datasets containing `-Xsqr-` are input for the batched matrix-inversion computational kernel of Bfast, which is implemented in folder `bfast-futhark/matrix-inv`. All other programs use the other datasets.

## Futhark Code Versions of Bfast (Whole) Program

Several GPU implementations can be derived from the futhark files *directly* located in the folder `bfast-futhark`:

* The main specification is in file `bfast-futhark\bfast.fut`. The most efficient implementation requires compilation under incremental flattening with the following threshold parameters:

---
$ FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend opencl --pass-option --default-tile-size=16 --pass-option --size=main.suff_outer_par_6=50000000 --pass-option --size=main.suff_intra_par_7=2048 --pass-option --size=main.suff_outer_par_8=50000000 --pass-option --size=main.suff_intra_par_9=2048 --pass-option --size=main.suff_outer_par_10=1  --pass-option --size=main.suff_intra_par_11=2048 --pass-option --size=main.suff_intra_par_13=1 --pass-option --size=main.suff_outer_par_17=50000000 --pass-option --size=main.suff_intra_par_18=2048 --pass-option --size=main.suff_outer_par_19=1 --pass-option --size=main.suff_intra_par_20=2048 --pass-option --size=main.suff_outer_par_21=50000000 --pass-option --size=main.suff_intra_par_22=2048 --pass-option --size=main.suff_outer_par_23=50000000 --pass-option --size=main.suff_intra_par_24=2048 --pass-option --size=main.suff_outer_par_25=50000000 --pass-option --size=main.suff_intra_par_26=2048 --pass-option --size=main.suff_outer_par_27=1 --pass-option --size=main.suff_intra_par_28=2048 --pass-option --size=main.suff_outer_par_29=50000000 --pass-option --size=main.suff_intra_par_30=1 --pass-option --size=main.suff_outer_par_33=50000000 --pass-option --size=main.suff_intra_par_34=1 --pass-option --size=main.suff_outer_par_35=50000000 --pass-option --size=main.suff_intra_par_36=2048 --pass-option --size=main.suff_outer_par_38=50000000 --pass-option --size=main.suff_intra_par_39=1 bfast.fut
---

* The same file `bfast-futhark/bfast.fut` compiled under moderate flattening, produces a version of the code that benefits from register+block tiling of the batched-matrix-matrix multiplication but does NOT utilizes intra-group parallelism in local memory. This version can be used to demonstrate the impact of local-memory usage at whole-application level. This version can be obtained by compiling with:

---
$ futhark bench --backend opencl --pass-option --default-tile-size=16 bfast.fut
---

* File `bfast-futhark/bfast-unoptim.fut`, compiled under moderate flattening, creates a GPU code version in which batched matrix-matrix multiplication used block tiling (but NOT register tiling) and which otherwise is NOT using local memory. This can be used to demonstrate the contributions of this paper. This code version can be obtained by compiling with:

---
$ futhark bench --backend opencl --pass-option --default-tile-size=16 bfast-unopt.fut
---

* File `bfast-futhark/bfast-fused.fut`, compiled under moderate flattening, creates a "naive" version of the code that fuses aggressively much of the computation, including all matrix-matrix multiplication. It is supposed to illustrate how slow a naive implementation can get, even when all global-memory accesses are coalesced. This code version can be obtained by compiling with:

---
$ futhark bench --backend opencl --pass-option --default-tile-size=16 bfast-fused.fut
---

## Futhark Code Version of Specific Computational Kernels of Bfast

Finally, in folder `bfast-futhark/indiv-kernels/` contains two illustrative computational kernels of Bfast, and is aimed at demonstrating the "local" impact, i.e., at the level of that computational kernel rather than at the application level.

* `bfast-futhark/indiv-kernels/batch-mmm` contains three different versions of batched-matrix-matrix multiplication: one by register+block tiling (`bmmm-regtile.fut`), one by only block tiling (`bmmm-blktile.fut`), and a "naive" one that performs no tiling at all (`bmmm-unopt.fut`).  For the version of the code that does block tiling, it is important to set the default tile size to `8`, otherwise it is very slow. For the other two, tile size does not matter.

---
$ futhark bench --backend opencl --pass-option --default-tile-size=8 bmmm-blktile.fut
---

* `bfast-futhark/indiv-kernels/matrix-inv` contains currently two versions of the code that can be both derived from file `matinv.fut`. Compiling with moderate flattening generate an all-flat version executing only in global memory. Compiling with incremental flattening and a specific threshold (see command below), generates an efficient version that executes mostly in local memory. Finally it would be nice to have also a version that exploits only the outermost parallelism but executes in global memory. This has been attempted in file `matinv-outer.fut`, but it does NOT work due to a compiler bug.

---
`$ FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend opencl --pass-option --size=main.suff_intra_par_1=1 matinv.fut`
---

## (II) C + OpenMP Code for Bfast (Whole) Program

This is implemented in folder `bfast-c`. The program creates its dataset internally, and requires the same parameters as `gen-data.fut`, namely, in order:

* an integer denoting the number of pixels in the image (`M`),
* an integer denoting the length of the data series (`N`),
* an integer denoting the number of pixels used for training (`n`),
* a float denoting the fraction of `NaN` values (`nanfreq`).
