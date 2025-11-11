.PHONY: venv install index run doctor render-up docker-up
venv:
	python -m venv .venv
install:
	. .venv/bin/activate && pip install -r requirements.txt
index:
	. .venv/bin/activate && python index_drive.py
run:
	. .venv/bin/activate && python app.py
doctor:
	. .venv/bin/activate && python doctor.py
render-up:
	chmod +x boot.sh
docker-up:
	docker compose up -d --build
