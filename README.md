# CSE 151B Competition — Team fork bomb

This repository contains the code for our submission to the CSE 151B/251B mathematical reasoning competition. Our approach focuses on maximizing the performance of a small language model, Alibaba's Qwen3-4B, through advanced prompt engineering and self-consistency techniques, as detailed in our final report.

**Final Result: We achieved an accuracy of 0.633 on the private test set, placing 21st out of 105 teams in the Kaggle competition.**

## Core Methodology

Our strategy was centered on optimizing inference-time behavior rather than fine-tuning the model's weights. We found that for a small model and a limited dataset (~1000 examples), prompt-based methods were significantly more effective and computationally cheaper than parameter-efficient fine-tuning (PEFT) like LoRA or QLoRA, which drastically degraded performance in our experiments.

Our final pipeline, implemented in `submit.py`, consists of three key components:

1.  **Advanced Prompt Engineering**: We developed highly specific prompts to guide the model's reasoning process and output format.
    *   **Role-Playing**: The model is instructed to act as an "expert mathematician."
    *   **Few-Shot Examples**: The prompt includes two examples (one MCQ, one free-form) to provide a clear template for the desired reasoning path and answer format.
    *   **Strict Output Control**: The system prompt explicitly commands the model to place *only* the final answer in a `\boxed{}` block at the very end of its response. This greatly improved the reliability of our `extract_last_boxed` parsing function.

2.  **Self-Consistency with Majority Voting**: To mitigate reasoning errors and improve robustness, we employed a self-consistency strategy, implemented in the `get_majority_voted_response` function.
    *   For each question, we generate **5 independent responses** (`n=5`) using the same prompt but with stochastic sampling (temperature = 0.6).
    *   We extract the `\boxed{}` answer from each of the 5 generations.
    *   The final answer is determined by a **majority vote** among the extracted candidates. The full text from the first generation that produced the majority answer is used as the final response.
    *   **Fallback**: In the rare case that no generation contains a `\boxed{}` answer, the system falls back to selecting the response with the highest overall log-probability.

3.  **Efficient Inference with vLLM**: The entire inference process is optimized using the `vLLM` library, enabling high-throughput generation on a single GPU.

## Code Implementation (`submit.py`)

The final submission logic is consolidated into a single script, `submit.py`. It is structured as follows:
1.  **Configuration**: Global constants for model ID, file paths, and generation hyperparameters.
2.  **Prompts & Examples**: The system prompts and few-shot examples are defined as string constants.
3.  **Helper Functions**:
    *   `build_prompt`: Constructs the full prompt for a given question.
    *   `extract_last_boxed`: A robust parser that finds the content of the last `\boxed{}` command, correctly handling nested braces.
    *   `get_majority_voted_response`: Implements the self-consistency and majority voting logic, including a fallback to the highest log-probability response if no boxed answers are found.
4.  **Main Inference Pipeline (`run_inference`)**:
    *   Initializes the `vLLM` engine and tokenizer.
    *   Loads the dataset from `data/private.jsonl`.
    *   Builds prompts for all questions.
    *   Runs batched inference to generate 5 responses per prompt.
    *   Processes the outputs using majority voting.
    *   Saves the final results to `results/submission.csv` and a detailed log to `results/starter_results.jsonl`.

## Key Hyperparameters

The following hyperparameters were used for the final submission, configured in `submit.py`:

| Parameter              | Value                         | Description                               |
| ---------------------- | ----------------------------- | ----------------------------------------- |
| `MODEL_ID`             | `Qwen/Qwen3-4B-Thinking-2507` | The base model used for inference.        |
| `NUM_GENERATIONS` (`n`) | `5`                           | Number of responses per question for self-consistency. |
| `TEMPERATURE`          | `0.6`                         | Controls the randomness of the output.    |
| `TOP_P`                | `0.95`                        | Nucleus sampling parameter.               |
| `TOP_K`                | `20`                          | Top-k sampling parameter.                 |
| `MAX_NEW_TOKENS`       | `38912`                       | Maximum number of tokens to generate.     |

## Execution Environment

*   **Model:** `Qwen/Qwen3-4B-Thinking-2507`
*   **GPU:** NVIDIA L40S
*   **Total Inference Time:** Approximately 12 hours to generate 5 responses for each of the 963 questions in the private dataset.
*   The final submission file generated by our code is available at `results/submission.csv`.

## Setup

### Dependencies
Before running, please install the necessary Python libraries. The core dependencies are listed below:
```bash
pip install sympy numpy transformers==4.57.6 vllm==0.19.1 tqdm bitsandbytes==0.45.5 triton antlr4-python3-runtime==4.11.1 ipykernel jupyter accelerate
```

### Model Weights
The script uses the `Qwen/Qwen3-4B-Thinking-2507` model, which is fetched directly from the Hugging Face Hub. The `vLLM` and `transformers` libraries will automatically download and cache the model the first time the script is run. No manual setup of model weights is required.

## How to Run Inference

To reproduce our final submission, execute the `submit.py` script from the root of the repository.

```bash
python submit.py
```

This script will:
1.  Load the model and tokenizer.
2.  Read the input data from `data/private.jsonl`.
3.  Generate responses for all questions.
4.  Save the final outputs to `results/submission.csv` and `results/starter_results.jsonl`.