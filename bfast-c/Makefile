PROGRAMS=bfast
INCLUDES=bfast.h

# Generic Makefile definitions.  Import from a Makefile, and be
# careful if changing it here.
# Can be run for example with 
# $ ./bfast 67968 414 228 0.0

CC=g++
CFLAGS?=-O2 -fopenmp
LDFLAGS = -lm

all: $(PROGRAMS)

%: %.c $(INCLUDES)
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.o
