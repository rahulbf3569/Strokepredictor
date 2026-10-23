.PHONY: setup data train test api app docker

setup:
	pip install -r requirements.txt

data:
	python data/generate_sample.py

train:
	python scripts/train.py

test:
	pytest -q

api:
	uvicorn api.main:app --reload --port 8000

app:
	streamlit run app/streamlit_app.py

docker:
	docker build -t stroke-risk-mlops . && docker run -p 8000:8000 stroke-risk-mlops