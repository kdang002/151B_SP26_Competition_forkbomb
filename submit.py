import json
import os
import re
import csv
from pathlib import Path
from typing import Optional, List
from collections import Counter
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    
DATA_PATH   = "data/private.jsonl"
OUTPUT_JSONL = "results/starter_results.jsonl"
OUTPUT_CSV  = "results/submission.csv"

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# Generation Hyperparameters (Matching Notebook)
NUM_GENERATIONS = 5
MAX_NEW_TOKENS  = 38912
TEMPERATURE     = 0.6
TOP_P           = 0.95
TOP_K           = 20
REPETITION_PENALTY = 1.0
MAX_MODEL_LEN = 16384
GPU_MEMORY_UTILIZATION = 0.95
MAX_NUM_SEQS = 32
MAX_NUM_BATCHED_TOKENS = 131072

# ==========================================
# 2. PROMPTS & EXAMPLES
# ==========================================
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician."
    "Give EXACT answers where possible: fractions or exact expressions like (1/2)^(36/31), not decimals. "
    "If decimals are required, use as many significant figures as possible, never round to fewer. "
    "Keep ALL intermediate calculations to as many significant figures as possible to avoid rounding errors. "
    "COUNT how many values the question asks for and put ALL of them in ONE \\\\boxed{}. "
    "For example, if the answer is 3.14159 and 2.71828, write \\\\boxed{3.14159, 2.71828}. "
    "NEVER put intermediate results in \\\\boxed{}. "
    "Only one \\\\boxed{} in your entire response, at the very end."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician."
    "Analyze the question and reason step-by-step."
    "Choose the correct option based on your reasoning."
    "Your final answer must be only \\\\boxed{X} where X is the option letter. "
    "Do not second-guess your answer. "
    "Your final answer must contain exactly one \\boxed{X} at the very end of the response, where X is the single option letter"
)

EXAMPLES = """Example 1 (MCQ)
Q: Let $f(x) = x^2 - 2x + 1$. What is the value of $f(3)$?
A. 2
B. 4
C. 6
D. 8
Answer: The function is $f(x) = x^2 - 2x + 1$.
We need to evaluate $f(3)$.
$f(3) = 3^2 - 2(3) + 1 = 9 - 6 + 1 = 4$.
The correct option is B.
\\\\boxed{B}

Example 2 (Free-form)
Q: A farm has chickens and rabbits. There are 35 heads and 94 feet in total. How many chickens and how many rabbits are on the farm?
Solution: Let c be the number of chickens and r be the number of rabbits.
Each animal has 1 head, so c + r = 35.
Chickens have 2 feet and rabbits have 4 feet, so 2c + 4r = 94.
From the first equation, c = 35 - r.
Substitute this into the second equation:
2(35 - r) + 4r = 94
70 - 2r + 4r = 94
2r = 24
r = 12
Now find c:
c = 35 - 12 = 23.
So there are 23 chickens and 12 rabbits. The question asks for the number of chickens and rabbits.
Final answer: \\\\boxed{23, 12}
"""

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Constructs the system and user prompts depending on question type."""
    examples_text = EXAMPLES
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user = examples_text + "\\n\\n" + f"{question}\\n\\nOptions:\\n{opts_text}"
        return SYSTEM_PROMPT_MCQ, user
    user = examples_text + "\\n\\n" + question
    return SYSTEM_PROMPT_MATH, user

def extract_last_boxed(text: str) -> Optional[str]:
    """Extract last \\\\boxed{} content, handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        if text[i:i+7] == r'\\boxed{':
            depth = 0
            start = i + 7
            j = start
            while j < len(text):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    if depth == 0:
                        results.append(text[start:j])
                        break
                    depth -= 1
                j += 1
        i += 1
    return results[-1].strip() if results else None

def get_majority_voted_response(output) -> str:
    """
    Extract the boxed answer from each generation, find the majority vote,
    and return the full text of the first generation that produced that answer.
    """
    def clean_text(text):
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    candidates = []
    for generation in output.outputs:
        cleaned = clean_text(generation.text)
        boxed_answer = extract_last_boxed(cleaned)
        if boxed_answer is not None:
            candidates.append((boxed_answer, generation.text.strip()))

    if not candidates:
        # Fallback to the highest logprob response if no boxed answers are found
        def score(generation):
            if not generation.logprobs:
                return float('-inf')
            return sum(max(token.logprob for token in step.values()) for step in generation.logprobs)

        best_generation = max(output.outputs, key=score)
        return best_generation.text.strip()

    answer_counts = Counter(ans for ans, txt in candidates)
    most_common_answer = answer_counts.most_common(1)[0][0]
    
    # Return the full text of the first response that gave the majority answer
    for ans, txt in candidates:
        if ans == most_common_answer:
            return txt
            
    return candidates[0][1]

# ==========================================
# 4. MAIN INFERENCE PIPELINE
# ==========================================
def run_inference():
    """
    Single entry point for loading the model, processing the dataset,
    running self-consistency inference, and outputting the CSV.
    """
    print(f"Loading Tokenizer and Model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        enable_prefix_caching=False,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    )

    sampling_params = SamplingParams(
        n=NUM_GENERATIONS,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=REPETITION_PENALTY,
        logprobs=1,
    )

    # Load Data
    print(f"Loading dataset from {DATA_PATH}")
    with open(DATA_PATH, "r") as f:
        data = [json.loads(line) for line in f]

    results = []

    print("Running inference...")
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt_text)

    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    results = []
    for item, output in tqdm(list(zip(data, outputs)), total=len(outputs), desc="Processing Questions"):
        # 1. Build Prompt
        # 2. Post-processing: Majority Voting
        final_response = get_majority_voted_response(output)
        
        results.append({
            "id": item.get("id"),
            "response": final_response
        })

    # 5. Save Final Submission JSONL and CSV
    jsonl_path = Path(OUTPUT_JSONL)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    submission_path = Path(OUTPUT_CSV)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(submission_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for r in results:
            writer.writerow([r["id"], r["response"]])

    print(f"Saved results JSONL to {jsonl_path}")
    print(f"Saved submission CSV to {submission_path}")

if __name__ == "__main__":
    run_inference()