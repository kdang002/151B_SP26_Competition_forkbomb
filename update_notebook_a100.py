import json
import re

with open("151b_model_google_colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    
    source = "".join(cell.get("source", []))
    original_source = source
    
    # Update Cell 1: git clone and chdir
    if "!git clone https://github.com/kdang002/151B_SP26_Competition_forkbomb.git" in source and "os.chdir(" in source:
        source = """# Cell 1: Clone repo and install packages
import os

repo_name = '151B_SP26_Competition_forkbomb'
if os.path.basename(os.getcwd()) != repo_name:
    if not os.path.exists(repo_name):
        !git clone https://github.com/kdang002/151B_SP26_Competition_forkbomb.git
    os.chdir(repo_name)

# Install dependencies
!pip install sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 accelerate scikit-learn peft trl datasets
"""

    # Update Cell 2: add PatchedStdout before vllm import
    if "# Cell 2: Setup configuration" in source and "from vllm import LLM, SamplingParams" in source:
        if "PatchedStdout" not in source:
            source = source.replace("from vllm import LLM, SamplingParams", """import io

# Patch the Jupyter stdout so fileno() doesn't crash vllm import
class PatchedStdout:
    def __init__(self, original):
        self._original = original
    def fileno(self):
        return 1  # fake a real file descriptor
    def __getattr__(self, name):
        return getattr(self._original, name)

if not isinstance(sys.stdout, PatchedStdout):
    sys.stdout = PatchedStdout(sys.stdout)

from vllm import LLM, SamplingParams""")
            
    # Update Cell 10 (or wherever PatchedStdout was duplicated)
    if "class PatchedStdout:" in source and "# Now load vLLM" in source:
        # We moved this to Cell 2. So we can just clear it or make it empty
        source = "# The stdout patch was moved to Cell 2 to prevent vllm import crash early on.\n"
        
    # Update BitsAndBytesConfig
    if "bnb_4bit_compute_dtype=torch.float16" in source:
        source = source.replace("bnb_4bit_compute_dtype=torch.float16", "bnb_4bit_compute_dtype=torch.bfloat16")
        
    # Update TrainingArguments
    if "fp16=True," in source:
        source = source.replace("fp16=True,", "bf16=True,")
        
    # Update vLLM init
    if "llm = LLM(" in source and "dtype=\"float16\"" in source:
        source = source.replace("dtype=\"float16\",", "dtype=\"bfloat16\",")
        
    if "gpu_memory_utilization=0.50" in source:
        source = source.replace("gpu_memory_utilization=0.50", "gpu_memory_utilization=0.85")
        
    if source != original_source:
        cell["source"] = [line + ("\n" if not line.endswith("\n") else "") for line in source.splitlines()]
        # remove trailing newline from last line to keep json format consistent
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open("151b_model_google_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated for A100 successfully.")
