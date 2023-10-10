DEVICE?=\#0 # If you wish, set the OpenCL device to use.
DATASETS=D1 D2 D3 D4 D5 D6 peru africa
GPU_RUNS=600
all: bfast-ours.json bfast-RegTl-EfSeq.json bfast-BlkTl-EfSeq.json bfast-All-EfSeq.json $(DATASETS:=.ops)

FUTHARK=/home/oscar/repositories/futhark/dist-newstyle/build/x86_64-linux/ghc-9.2.8/futhark-0.26.0/x/futhark/build/futhark/futhark

bfast-ours.json: bfast-ours.fut
	$(FUTHARK) bench --backend cuda --pass-option --default-tile-size=8 $< --json $@ --runs $(GPU_RUNS) --futhark=$(FUTHARK)

bfast-RegTl-EfSeq.json: bfast-RegTl-EfSeq.fut
	$(FUTHARK) bench --backend cuda --pass-option --default-tile-size=8 $< --json $@ --runs $(GPU_RUNS) --futhark=$(FUTHARK)

bfast-All-EfSeq.json: bfast-All-EfSeq.fut
	$(FUTHARK) bench --backend cuda --pass-option --default-tile-size=8 $< --json $@ --runs $(GPU_RUNS) --futhark=$(FUTHARK)

bfast-BlkTl-EfSeq.json: bfast-BlkTl-EfSeq.fut
	$(FUTHARK) bench --backend cuda --pass-option --default-tile-size=8 $< --json $@ --runs $(GPU_RUNS) --futhark=$(FUTHARK)

data/perf-calculator: data/perf-calculator.fut
	$(FUTHARK) c $<

%.ops: data/perf-calculator
	gunzip < data/$*.in.gz | data/perf-calculator | head -n 3 | sed s/u64// > $@

clean:
	rm -f *.json *.c bfast bfast-unopt bfast-fused *.ops data/perf-calculator
