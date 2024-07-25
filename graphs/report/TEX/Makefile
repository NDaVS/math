FILENAME = "one_file_report"
RCS_FILENAME = $(FILENAME).tex
NUWEB = nuweb
CI = ci
CO = co
RCSDIFF = rcsdiff
PDFLATEX = pdflatex
BIBTEX = bibtex

all: pdflatex-final

nuweb: 
	@echo "Running $(NUWEB) -c $(FILENAME)..."
	$(NUWEB) -c $(FILENAME).w 
	@if [ $$? -ne 0 ]; then \
		echo "Error: $(NUWEB) -c failed"; \
		exit 1; \
	else \
		echo "$(NUWEB) -c completed successfully."; \
	fi
	$(PDFLATEX) $(RCS_FILENAME) > pdflatex.log
	$(NUWEB) -c $(FILENAME).w

rcs: 
	@if [ ! -f "RCS/$(RCS_FILENAME),v" ]; then \
		echo "Adding $(RCS_FILENAME) to RCS..."; \
		$(CI) -l $(RCS_FILENAME); \
		if [ $$? -ne 0 ]; then \
			echo "Error: Failed to add $(RCS_FILENAME) to RCS"; \
			exit 1; \
		fi \
	else \
		echo "$(RCS_FILENAME) is already under RCS control."; \
	fi
	$(CO) -l $(RCS_FILENAME)

check-changes: 
	@echo "Checking for changes in $(RCS_FILENAME)..."
	@rcsdiff_output=`$(RCSDIFF) $(RCS_FILENAME)`; \
	if [ -n "$$rcsdiff_output" ]; then \
		echo "Changes detected. Committing new version..."; \
		$(CI) -u $(RCS_FILENAME); \
		if [ $$? -ne 0 ]; then \
			echo "Error: Failed to commit new version of $(RCS_FILENAME)"; \
			exit 1; \
		fi \
	else \
		echo "No changes detected in $(RCS_FILENAME)."; \
	fi

pdflatex: 
	@echo "Running $(PDFLATEX) $(FILENAME)..."
	$(PDFLATEX) $(FILENAME) > pdflatex.log
	@if [ $$? -ne 0 ]; then \
		echo "Error: $(PDFLATEX) failed"; \
		exit 1; \
	else \
		echo "$(PDFLATEX) completed successfully."; \
	fi
nuweb2: 
	@echo "Running $(NUWEB) -c $(FILENAME)..."
	$(NUWEB) -c $(FILENAME).w 
	@if [ $$? -ne 0 ]; then \
		echo "Error: $(NUWEB) -c failed"; \
		exit 1; \
	else \
		echo "$(NUWEB) -c completed successfully."; \
	fi

bibtex: 
	@echo "Running $(BIBTEX) $(FILENAME)..."
	$(BIBTEX) $(FILENAME) > bibtex.log
	@if [ $$? -ne 0 ]; then \
		echo "Error: $(BIBTEX) failed"; \
		exit 1; \
	else \
		echo "$(BIBTEX) completed successfully."; \
	fi

pdflatex-final: nuweb rcs check-changes pdflatex bibtex pdflatex pdflatex
	@echo "Running $(PDFLATEX) $(FILENAME) for the final time..."
	$(PDFLATEX) $(FILENAME) > pdflatex.log
	@if [ $$? -ne 0 ]; then \
		echo "Error: $(PDFLATEX) failed"; \
		exit 1; \
	else \
		echo "$(PDFLATEX) completed successfully."; \
	fi

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz $(FILENAME).pdf $(FILENAME).tex
