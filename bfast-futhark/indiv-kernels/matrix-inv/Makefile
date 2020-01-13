DEVICE?=\#0 # If you wish, set the OpenCL device to use.

FUTHARK=../../futhark-0.11.2-linux-x86_64/bin/futhark

CUDA=/usr/local/cuda-10.0/

all: matinv-glob-mem.json matinv-fast-mem.json $(DATASETS:=.ops)

matinv-glob-mem.json: matinv.fut
	$(FUTHARK) bench --json $@ --backend=opencl $< --runs 200 --pass-option=-d'$(DEVICE)' --futhark=$(FUTHARK)

matinv-fast-mem.json: matinv.fut
	FUTHARK_INCREMENTAL_FLATTENING=1 $(FUTHARK) bench --backend opencl --json $@ --pass-option --size=main.suff_intra_par_1=1 matinv.fut --runs 200 --pass-option=-d'$(DEVICE)' --futhark=$(FUTHARK)

matinv-cublas: mat-inv.cu
	$(CUDA)/bin/nvcc  -o matinv-cublas -I $(CUDA)/include -L $(CUDA)/lib64 mat-inv.cu -lcublas
	./matinv-cublas 16384 8
	./matinv-cublas 32768 8
	./matinv-cublas 65536 8
	./matinv-cublas 111556 8
