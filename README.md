# futhark-kdd19

##Bfast Performance Evaluation

Folder `bfast-c` contains a OpenMP-parallelized CPU version of bfast; the code was carefully written to be in a good form for vectorization, but it is ultimately up to gcc to extract it, i.e., no vectorization intrinsics have been used. The datasets are generated internally.

Folder `bfast-futhark` contain various bfast versions of the code written in Futhark. This includes individual compuational kernels of bfast including batched-matrix-matrix multiplication and batched matrix inversion. Also datasets and means of generating datasets are provided. Deatils are as follows:

### Datasets:

The datasets are located in `bfast-futhark/data`: dataset `sahara.in` and `peru.in` correspond to actual images taken from Sahara and Peru. There are a bunch of contrived datasets, for example `d-32768-256-128.in.gz` corresponds to a randomly generated image in which the image has `32768` pixels, the time-series length is `256` and half of the time series (`128`) is used for training. In all contrived datasets `NaN` values appear with a frequency of `50%`, but we should vary that a little bit to demonstrate that peformance is not influenced by the percentage of `NaN` values. 

The datasets have been generated with the futhark program `bfast-futhark/data/gen-datasets/gen-data.fut`, which receive as parameters in order: the number of pixels in the image (`M`), the length of the data series (`N`), the number of pixels used for training (`n`), and the fraction of `NaN` values (`nanfreq`). For example `nanfreq = 0.5` corresponds to half the values beeing `NaN`.

An estimator of the number of floating point operations (Flops) for a dataset is implemented in Futhark program `perf-calculator.fut`. This estimates the Flops as they appear in the `bfast.fut` program. Floating point operations include, addition, multiplication, division, casts, `isnan`, special functions (`sqrt`, `log`), etc. The input of `perf-calculator.fut` is any dataset of Bfast.

Finally, in folder `bfast-futhark/data`, the datasets containing `-Xsqr-` are input for the batched matrix-inversion computational kernel of Bfast, which is implemented in folder `bfast-futhark/matrix-inv`. All other programs use the other datasets.

## Bfast Code Versions (Full Program)

Several GPU implementations can be derived from the futhark files directly located in the folder `bfast-futhark`:

* The main specification is in file `bfast-futhark\bfast.fut`. At least two important GPU implementations can be derived from here. The most performant implementation requires compilation under incremental flattening:

--
FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend opencl --pass-option --default-tile-size=16 --pass-option --size=main.suff_outer_par_6=50000000 --pass-option --size=main.suff_intra_par_7=2048 --pass-option --size=main.suff_outer_par_8=50000000 --pass-option --size=main.suff_intra_par_9=2048 --pass-option --size=main.suff_outer_par_10=1  --pass-option --size=main.suff_intra_par_11=2048 --pass-option --size=main.suff_intra_par_13=1 --pass-option --size=main.suff_outer_par_17=50000000 --pass-option --size=main.suff_intra_par_18=2048 --pass-option --size=main.suff_outer_par_19=1 --pass-option --size=main.suff_intra_par_20=2048 --pass-option --size=main.suff_outer_par_21=50000000 --pass-option --size=main.suff_intra_par_22=2048 --pass-option --size=main.suff_outer_par_23=50000000 --pass-option --size=main.suff_intra_par_24=2048 --pass-option --size=main.suff_outer_par_25=50000000 --pass-option --size=main.suff_intra_par_26=2048 --pass-option --size=main.suff_outer_par_27=1 --pass-option --size=main.suff_intra_par_28=2048 --pass-option --size=main.suff_outer_par_29=50000000 --pass-option --size=main.suff_intra_par_30=1 --pass-option --size=main.suff_outer_par_33=50000000 --pass-option --size=main.suff_intra_par_34=1 --pass-option --size=main.suff_outer_par_35=50000000 --pass-option --size=main.suff_intra_par_36=2048 --pass-option --size=main.suff_outer_par_38=50000000 --pass-option --size=main.suff_intra_par_39=1 bfast.fut
--


