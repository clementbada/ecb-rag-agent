.PHONY: install test index eval-dev eval-test agent security ask app all

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

index:
	python scripts/build_index.py

eval-dev:
	python scripts/eval_retrieval.py --split dev

eval-test:
	python scripts/eval_retrieval.py --split test

agent:
	python scripts/eval_agent.py --split test --baseline

security:
	python scripts/eval_security.py

ask:
	python scripts/ask.py "$(Q)"

app:
	streamlit run app.py

all: test index eval-dev eval-test agent security
