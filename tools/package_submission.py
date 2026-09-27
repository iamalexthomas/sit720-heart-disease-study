"""Create an uploadable code/data ZIP with no virtual environment or caches."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT.parent / "SIT720_HD_Code_and_Data.zip"
TOP_LEVEL = ["README.md", ".gitignore", "requirements.txt",
             "run_experiments.py", "research_notebook.ipynb"]

with zipfile.ZipFile(DESTINATION, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in TOP_LEVEL:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Required deliverable missing: {path}")
        archive.write(path, f"hd_submission/{name}")
    for folder in ["data", "results", "report", "tools"]:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, str(Path("hd_submission") / path.relative_to(ROOT)))
print(f"Created {DESTINATION}")
