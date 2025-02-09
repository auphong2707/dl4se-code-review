import shutil
from huggingface_hub import snapshot_download
import os
from pathlib import Path

repo_id = "auphong2707/dl4se-code-review"
model_subdir = "code-summarization/code-t5-base/experiment-1"
local_dir = "./my_model/code-t5-base/experiment-1"
temp_dir = "./temp_hf_download"

# Download the model from the repo
Path(temp_dir).mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    revision="main",
    local_dir=temp_dir,
    allow_patterns=[model_subdir + '/*'],
    ignore_patterns=[model_subdir + '/checkpoint-*/*'],
)

# Remove cache files
shutil.rmtree(os.path.join(temp_dir, ".cache"))

# Flatten the directory and copy files to the target local_dir
local_dir = Path(local_dir)
local_dir.mkdir(parents=True, exist_ok=True)

for file in Path(temp_dir).glob("**/*"):
    if file.is_file():
        shutil.copy(file, local_dir / file.name)

# Cleanup temporary directory
shutil.rmtree(temp_dir)

print(f"Files from {repo_id}/{model_subdir} have been downloaded and saved to: {local_dir}")