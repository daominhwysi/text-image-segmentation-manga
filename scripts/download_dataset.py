import zipfile
import os
from roboflow import Roboflow
from huggingface_hub import hf_hub_download

rf = Roboflow(api_key="P3FPqjdT3M8cUVClBV2n")
project = rf.workspace("dao-minh-uyi1j").project("444-8vbul")
version = project.version(2)
dataset = version.download("yolov12")

file_path = hf_hub_download(
    repo_id="Daominhwysi/font-collection",
    filename="font-collection.zip",
    repo_type="dataset",
)

extraction_path = "./fonts"
with zipfile.ZipFile(file_path, "r") as zip_ref:
    zip_ref.extractall(extraction_path)

print(f"Fonts extracted to: {os.path.abspath(extraction_path)}")
