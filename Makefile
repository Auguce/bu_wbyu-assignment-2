# Makefile

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

run:
	@echo "Running the Flask application..."
	MPLCONFIGDIR=./ python app.py

