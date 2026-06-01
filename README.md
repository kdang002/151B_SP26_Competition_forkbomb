# CSE 151B Competition — Team fork bomb


## Execution Environment

*   **GPU:** NVIDIA L40S
*   **Total Inference Time:** Approximately 12 hours to generate results for the private dataset.
*   result is already in `results` folder. 

## Setup

### Dependencies
Before running, please install the necessary Python libraries. The core dependencies are listed below:
```bash
pip install sympy numpy transformers==4.57.6 vllm==0.19.1 tqdm bitsandbytes==0.45.5 triton antlr4-python3-runtime==4.11.1 ipykernel jupyter accelerate
```

### Model Weights
The script uses the `Qwen/Qwen3-4B-Thinking-2507` model, which is fetched directly from the Hugging Face Hub. The `vLLM` and `transformers` libraries will automatically download and cache the model the first time the script is run. No manual setup of model weights is required.

## How to Run Inference

To reproduce the results, you can call the `run_inference()` function by executing the `submit.py` script from your terminal.

```bash
python submit.py
```

This script will:
1.  Load the model and tokenizer.
2.  Read the input data from `data/private.jsonl`.
3.  Generate responses for all questions.
4.  Save the final outputs to `results/submission.csv` and `results/starter_results.jsonl`.