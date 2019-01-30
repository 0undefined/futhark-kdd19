DEVICE?=\#0 # If you wish, set the OpenCL device to use.

all: bmmm-blktile.json bmmm-regtile.json bmmm-unoptim.json

bmmm-%.json: bmmm-%.fut
	futhark bench --json $@ --backend=opencl $< --runs 100 --pass-option --default-tile-size=8 --pass-option=-d'$(DEVICE)'
