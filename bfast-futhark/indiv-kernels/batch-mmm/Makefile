DEVICE?=\#0 # If you wish, set the OpenCL device to use.

FUTHARK=../../futhark-0.11.2-linux-x86_64/bin/futhark

all: bmmm-blktile.json bmmm-regtile.json bmmm-unoptim.json $(DATASETS:=.ops)

bmmm-%.json: bmmm-%.fut
	$(FUTHARK) bench --json $@ --backend=opencl $< --runs 250 --pass-option --default-tile-size=8 --pass-option=-d'$(DEVICE)' --futhark=$(FUTHARK)
