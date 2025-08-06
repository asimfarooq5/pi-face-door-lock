install_uv:
	@if ! command -v uv >/dev/null 2>&1; then \
  		curl -LsSf https://astral.sh/uv/install.sh | sh; \
  	fi

setup:
	make install_uv
	uv venv
	uv pip install -r requirements.txt

lint:
	./.venv/bin/ruff format .

check-lint:
	./.venv/bin/ruff check .

dataset:
	python3 dataset_creator.py

model-train:
	python3 model_trainer.py

recognize:
	python3 face_recognizer.py

run:
	python3 run.py


