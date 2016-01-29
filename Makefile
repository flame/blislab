ARTICLE  = paper
all:
	pdflatex paper
	bibtex paper
	pdflatex paper
	pdflatex paper

clean:
	rm -R *~; rm *.bbl *.blg *.pdf *.ps *.dvi *.aux *.out; rm *.idx; rm *.log; rm *.toc; rm */*~

