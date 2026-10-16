.PHONY: install test lint evaluate app
install:
	pip install -e ".[app,dev]"
lint:
	ruff check src tests app
test:
	pytest -q
evaluate:
	python -m roomfinder.cli evaluate --data-dir data/raw --out reports
app:
	streamlit run app/streamlit_app.py
