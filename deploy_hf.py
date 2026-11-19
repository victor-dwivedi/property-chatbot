"""Deploy to Hugging Face Spaces: private dataset for the scraped data + public Docker Space for the app.
Reads keys from .env. Usage: .venv/bin/python deploy_hf.py
"""
import tempfile, shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
env = {}
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

user, token = env["HF_USERNAME"], env["HF_TOKEN"]
SPACE, DATA = f"{user}/property-chatbot", f"{user}/property-chatbot-data"
APP_URL = f"https://{user}-property-chatbot.hf.space"
api = HfApi(token=token)

# 1) private dataset with the scraped data
api.create_repo(DATA, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=DATA, repo_type="dataset", folder_path=str(ROOT / "data"),
                  allow_patterns=["*.jsonl"], commit_message="Update scraped data")
print("data uploaded ->", DATA, "(private)")

# 2) public Docker Space with the app
api.create_repo(SPACE, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
for k, v in {"OPENROUTER_API_KEY": env["OPENROUTER_API_KEY"], "HF_TOKEN": token}.items():
    api.add_space_secret(SPACE, k, v)
for k, v in {"DATA_REPO": DATA, "APP_URL": APP_URL}.items():
    api.add_space_variable(SPACE, k, v)

with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp)
    shutil.copytree(ROOT / "app", t / "app", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ["Dockerfile", "requirements.txt", ".dockerignore"]:
        shutil.copy(ROOT / f, t / f)
    (t / "README.md").write_text(
        "---\ntitle: DarGlobal & Wasalt Property Chatbot\nemoji: 🏠\ncolorFrom: yellow\ncolorTo: gray\n"
        "sdk: docker\napp_port: 7860\npinned: false\n---\n\n"
        + (ROOT / "README.md").read_text())
    api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=tmp, commit_message="Deploy chatbot")
print("space deployed ->", f"https://huggingface.co/spaces/{SPACE}")
print("app url ->", APP_URL)
