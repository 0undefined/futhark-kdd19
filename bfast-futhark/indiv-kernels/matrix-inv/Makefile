DEVICE?=\#0 # If you wish, set the OpenCL device to use.

all: matinv-glob-mem.json matinv-fast-mem.json $(DATASETS:=.ops)

matinv-glob-mem.json: matinv.fut
	futhark bench --json $@ --backend=opencl $< --runs 100 --pass-option=-d'$(DEVICE)'

matinv-fast-mem.json: matinv.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend opencl --json $@ --pass-option --size=main.suff_intra_par_1=1 matinv.fut --pass-option=-d'$(DEVICE)'
