import zipfile
import os
from roboflow import Roboflow
from huggingface_hub import hf_hub_download

base_dir = os.getcwd()
target_folder = os.path.join(base_dir, "resource")

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

os.chdir(target_folder)

rf = Roboflow(api_key="f01cbXEWTlXhJAuV2PVq")
project = rf.workspace("dao-minh").project("speech-bubble-ibar9")
version = project.version(6)
dataset = version.download("yolo26")

os.chdir(base_dir)
file_path = hf_hub_download(
    repo_id="Daominhwysi/font-collection",
    filename="font-collection.zip",
    repo_type="dataset",
)

extraction_path = os.path.join(target_folder, "fonts")
with zipfile.ZipFile(file_path, "r") as zip_ref:
    zip_ref.extractall(extraction_path)

print(f"Dataset location: {dataset.location}")
print(f"Fonts extracted to: {os.path.abspath(extraction_path)}")
