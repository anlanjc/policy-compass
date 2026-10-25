.PHONY: install ingest serve chat eval test docker clean

install:
	pip install -r requirements.txt

ingest:
	python scripts/ingest.py

serve:
	python scripts/serve.py

chat:
	@python scripts/chat.py "$(q)"

eval:
	python scripts/eval.py

test:
	python -m pytest tests/ -q

docker:
	docker compose up --build

clean:
	rm -rf data/index.db* data/traces docs/eval_report.md
