PROGRAMS=bfast
INCLUDES=bfast.h
DATASETS=D1 D2 D3 D4 D5 D6 peru

# Generic Makefile definitions.  Import from a Makefile, and be
# careful if changing it here.
# Can be run for example with 
# $ ./bfast 67968 414 228 0.5

CC?=gcc
CFLAGS?=-O3 -fopenmp -std=c99
LDFLAGS = -lm

all: $(PROGRAMS) $(DATASETS:=.runtime)

%: %.c $(INCLUDES)
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

D1.runtime:
	./bfast 16384 1024 512 0.5 | tee /dev/stdout | tail -n 1 | sed s/μs// > $@

D2.runtime:
	./bfast 16384 512 256 0.5 | tee /dev/stdout | tail -n 1 | sed s/μs// > $@

D3.runtime:
	./bfast 32768 512 256 0.5 | tee /dev/stdout | tail -n 1 | sed s/μs// > $@

D4.runtime:
	./bfast 32768 256 128 0.5 | tee /dev/stdout | tail -n 1 | sed s/μs// > $@

D5.runtime:
	./bfast 65536 256 128 0.5 | tee /dev/stdout | tail -n 1 | sed s/μs// > $@

D6.runtime:
	./bfast 16384 1024 256 0.75 | tee /dev/stdout | tail -n 1 | sed s/μs// > $@

peru.runtime:
	./bfast 111556 235 113 0.689226 | tee /dev/stdout | tail -n 1 | sed s/μs// > $@

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.o
