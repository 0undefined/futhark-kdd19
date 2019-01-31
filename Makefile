all: batch-mmm.pdf matinv.pdf bfast.pdf

%.pdf: plot-%.py
	python plot-$*.py $@

clean:
	rm -f *.pdf
