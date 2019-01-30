DEVICE?=\#0 # If you wish, set the OpenCL device to use.
DATASETS=sahara peru d-16384-1024-512 d-16384-512-256 d-32768-256-128 d-32768-512-256 d-65536-128-64 d-65536-256-128

all: bfast.json bfast-moderate.json bfast-unopt.json bfast-fused.json $(DATASETS:=.ops)

bfast.json: bfast.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend opencl --pass-option --default-tile-size=16 --pass-option --size=main.suff_outer_par_6=50000000 --pass-option --size=main.suff_intra_par_7=2048 --pass-option --size=main.suff_outer_par_8=50000000 --pass-option --size=main.suff_intra_par_9=2048 --pass-option --size=main.suff_outer_par_10=1  --pass-option --size=main.suff_intra_par_11=2048 --pass-option --size=main.suff_intra_par_13=1 --pass-option --size=main.suff_outer_par_17=50000000 --pass-option --size=main.suff_intra_par_18=2048 --pass-option --size=main.suff_outer_par_19=1 --pass-option --size=main.suff_intra_par_20=2048 --pass-option --size=main.suff_outer_par_21=50000000 --pass-option --size=main.suff_intra_par_22=2048 --pass-option --size=main.suff_outer_par_23=50000000 --pass-option --size=main.suff_intra_par_24=2048 --pass-option --size=main.suff_outer_par_25=50000000 --pass-option --size=main.suff_intra_par_26=2048 --pass-option --size=main.suff_outer_par_27=1 --pass-option --size=main.suff_intra_par_28=2048 --pass-option --size=main.suff_outer_par_29=50000000 --pass-option --size=main.suff_intra_par_30=1 --pass-option --size=main.suff_outer_par_33=50000000 --pass-option --size=main.suff_intra_par_34=1 --pass-option --size=main.suff_outer_par_35=50000000 --pass-option --size=main.suff_intra_par_36=2048 --pass-option --size=main.suff_outer_par_38=50000000 --pass-option --size=main.suff_intra_par_39=1 bfast.fut --json $@ --runs 100 --pass-option=-d'$(DEVICE)'

bfast-moderate.json: bfast.fut
	futhark bench --backend opencl --pass-option --default-tile-size=16 bfast.fut --json $@ --runs 100 --pass-option=-d'$(DEVICE)'

bfast-unopt.json: bfast-unopt.fut
	futhark bench --backend opencl --pass-option --default-tile-size=16 bfast-unopt.fut --json $@ --runs 100 --pass-option=-d'$(DEVICE)'

bfast-fused.json: bfast-fused.fut
	futhark bench --backend opencl --pass-option --default-tile-size=16 bfast-fused.fut --json $@ --runs 10 --pass-option=-d'$(DEVICE)'

data/perf-calculator: data/perf-calculator.fut
	futhark c $<

%.ops: data/perf-calculator
	gunzip < data/$*.in.gz | data/perf-calculator | head -n 3 | sed s/u64// > $@

clean:
	rm -f *.json *.c bfast bfast-unopt bfast-fused *.ops data/perf-calculator
