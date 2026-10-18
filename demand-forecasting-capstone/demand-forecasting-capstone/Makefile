.PHONY: install build eda run test dashboard

install:
	pip install -e ".[dev,dashboard]"

build:
	demand-agents build --config synthetic

eda:
	demand-agents eda --config synthetic

run:
	demand-agents run --config synthetic

test:
	pytest -q

dashboard:
	streamlit run dashboard/app.py
