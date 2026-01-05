First, Download the dataset and fonts
`python scripts/download_dataset.py`

Then you can generate dataset `python -m src.generator`

login hf:
python -c "from huggingface_hub import notebook_login; notebook_login()"
upload dataset:
python scripts/upload_dataset.py
