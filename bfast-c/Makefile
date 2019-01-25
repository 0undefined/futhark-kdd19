PROGRAMS=bfast
INCLUDES=bfast.h
DATASETS=peru d-16384-1024-512 d-16384-512-256 d-32768-256-128 d-32768-512-256 d-65536-128-64 d-65536-256-128

# Generic Makefile definitions.  Import from a Makefile, and be
# careful if changing it here.
# Can be run for example with 
# $ ./bfast 67968 414 228 0.5

CC?=gcc
CFLAGS?=-O3 -fopenmp
LDFLAGS = -lm

all: $(PROGRAMS) $(DATASETS:=.runtime)

%: %.c $(INCLUDES)
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

peru.runtime:
	./bfast 111556 235 113 0.5 | tee /dev/stdout | tail -n 1 | sed s/μs// > $@

d-%.runtime:
	./bfast $$(echo $* | sed 's/-/ /g') 0.5 | tee /dev/stdout | tail -n 1 | sed s/μs// > $@

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.o
