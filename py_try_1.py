# %% [markdown]
# ## First run this in dsmlp 
# launch-sp26-cuda128.sh -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 8 -m 32 -v a30
# ##
# And check you have enough space to run the model from root dir

# %% [markdown]
# # CSE 151B Competition — Starter Notebook
# 
# Welcome to the **CSE 151B Spring 2026 Math Reasoning Competition**!  
# This notebook walks you through the full pipeline end-to-end:
# 
# 1. Setting up the Python environment with `uv`
# 2. Loading the competition dataset
# 3. Running inference with **Qwen3-4B-Thinking** via Transformer (INT8 quantized)
# 4. NOTE: DO NOT RUN ON vLLM. The team has notorious trouble with vLLM. ONLY RUN WITH Transformer.
# 5. Scoring responses against ground-truth answers
# 6. Saving results to JSONL for submission
# 
# The public dataset (`public.jsonl`) contains questions **with** answers so you can measure accuracy locally.  
# The private test set used for the leaderboard does **not** include answers — for that, skip evaluation and submit the raw responses.

# %% [markdown]
# ## 1. Import modules and packages
# 
# ## 2. Imports & Configuration
# 
# All key settings are collected in one place.  
# - `DATA_PATH` — public dataset with ground-truth answers (use this to measure accuracy)
# - `OUTPUT_PATH` — where per-question results will be written
# - `GPU_ID` — which GPU to use (update if your machine has a different device index)
# - `MAX_TOKENS` — maximum tokens the model may generate per response

# %%
#you might need to change DATA_PATH, OUTPUT_PATH
import json
import os

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/starter_results.jsonl"
MAX_TOKENS  = 32768         

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
import sys
from pathlib import Path
from typing import Optional
from fractions import Fraction


from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from collections import Counter

# %%
#check what gpu you have and gpu id number
#!nvidia-smi

# %% [markdown]
# ## 3. Load the Dataset
# 
# The dataset is stored as newline-delimited JSON (`.jsonl`). Each line is one question with the following fields:
# 
# | Field | Description |
# |---|---|
# | `id` | Unique question identifier |
# | `question` | Problem statement |
# | `options` | List of answer choices — present for **MCQ**, absent for **free-form** |
# | `answer` | Ground-truth answer (letter for MCQ, value/list for free-form) |

# %%
data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

# Preview one MCQ and one free-form item
mcq_sample  = next(d for d in data if d.get("options"))
free_sample = next(d for d in data if not d.get("options"))

print("\n── MCQ sample ──")
print(json.dumps(mcq_sample, indent=2))
print("\n── Free-form sample ──")
print(json.dumps(free_sample, indent=2))

# %% [markdown]
# ## 4. Prompt Construction
# 
# We use two system prompts depending on the question type:
# 
# - **MCQ** — the model must select the best answer letter and wrap it in `\boxed{}`
# - **Free-form** — the model solves step-by-step and puts the final answer in `\boxed{}`
# 
# ### Also implementing Fewshots
# 
# `build_prompt()` returns the appropriate `(system, user)` pair for each item.

# %%
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. "
    "Give EXACT answers where possible: fractions or exact expressions like (1/2)^(36/31), not decimals. "
    "If decimals are required, use as many significant figures as possible, never round to fewer. "
    "Keep ALL intermediate calculations to as many significant figures as possible to avoid rounding errors. "
    "COUNT how many values the question asks for and put ALL of them in ONE \\\\boxed{}. "
    "For example, if the answer is 3.14159 and 2.71828, write \\\\boxed{3.14159, 2.71828}. "
    "NEVER put intermediate results in \\\\boxed{}. "
    "Only one \\\\boxed{} in your entire response, at the very end."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Choose the correct option. "
    "Your final answer must be only \\\\boxed{X} where X is the option letter. "
    "Do not second-guess your answer. "
    "Only one \\\\boxed{} in your response."
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

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    examples_text = EXAMPLES
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user = examples_text + "\\n\\n" + f"{question}\\n\\nOptions:\\n{opts_text}"
        return SYSTEM_PROMPT_MCQ, user
    user = examples_text + "\\n\\n" + question
    return SYSTEM_PROMPT_MATH, user

def extract_last_boxed(text: str) -> Optional[str]:
    """Extract last \\\\boxed{} content, handling nested braces like \\\\frac{5}{8}"""
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


# Verify with samples
for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
    sys_p, usr_p = build_prompt(item["question"], item.get("options"))
    print(f"── {label} user prompt (first 200 chars) ──")
    print(usr_p[:200], "...\\n")


# %% [markdown]
# ## 5. Load Model with vLLM (for general case, vLLM is faster)
# 
# 
# We load **Qwen3-4B-Thinking-2507** with **INT8 quantization** via BitsAndBytes.  
# Setting `load_format="bitsandbytes"` tells vLLM to apply on-the-fly INT8 weight quantization, roughly halving GPU memory usage compared to BF16.
# 
# Key parameters:
# - `gpu_memory_utilization` — fraction of GPU VRAM reserved for the model and KV cache
# - `max_model_len` — maximum sequence length (prompt + generation)
# - `max_num_seqs` — maximum number of sequences processed in parallel

# %%
# import math
# from tqdm.notebook import tqdm # Use tqdm.notebook for Jupyter environments

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# print("--- Calculating optimal vLLM parameters ---")

# # --- Step 1: Count the maximum input tokens observed across your dataset ---
# max_input_tokens_seen = 0
# print(f"Analyzing {len(data)} items to determine max input token length...")

# # Iterate through your entire dataset to find the prompt with the most tokens.
# # We're using the same prompt building logic as your inference loop.
# for item in tqdm(data, desc="Tokenizing prompts"):
#     system, user = build_prompt(item["question"], item.get("options"))

#     prompt_text = tokenizer.apply_chat_template(
#         [{"role": "system", "content": system},
#          {"role": "user",   "content": user}],
#         tokenize=False,
#         add_generation_prompt=True,
#     )

#     current_input_tokens = len(tokenizer.encode(prompt_text))

#     if current_input_tokens > max_input_tokens_seen:
#         max_input_tokens_seen = current_input_tokens

# print(f"\nMaximum input tokens observed (including chat template): {max_input_tokens_seen}")

# # --- Step 2: Define your desired maximum output tokens (from sampling_params) ---
# desired_max_output_tokens = 4096 
# print(f"Desired maximum output tokens per generation: {desired_max_output_tokens}")

# # --- Step 3: Calculate max_model_len for the LLM instance ---
# BUFFER_TOKENS = 128 
# calculated_max_model_len = max_input_tokens_seen + desired_max_output_tokens + BUFFER_TOKENS
# max_model_len_final = max(512, calculated_max_model_len)
# print(f"Calculated max_model_len for LLM: {max_model_len_final} (from {max_input_tokens_seen} input + {desired_max_output_tokens} output + {BUFFER_TOKENS} buffer)")

# # --- Step 4: Determine max_num_seqs ---
# num_generations_free = 4
# num_generations_mcq = 4
# max_num_seqs_final = max(num_generations_free, num_generations_mcq)
# print(
#     f"Setting max_num_seqs to: {max_num_seqs_final} "
#     f"(free-form n={num_generations_free}, mcq n={num_generations_mcq})"
# )

# # --- Step 5: Set max_num_batched_tokens ---
# initial_batch_capacity_estimate = max_input_tokens_seen * max_num_seqs_final
# max_num_batched_tokens_final = max(32768, initial_batch_capacity_estimate * 2) 
# print(f"Initial estimate for max_num_batched_tokens: {max_num_batched_tokens_final}. "
#       "This value is heuristic and might require empirical tuning.")


# # --- Step 6: Configure the LLM instance with the calculated parameters ---
# print("\n--- Configuring LLM with optimized parameters ---")
llm = LLM(
    model=MODEL_ID,
    #quantization="bitsandbytes", # Using bfloat16 for better compatibility with Ampere GPUs (A30)
    #load_format="bitsandbytes",
    dtype="bfloat16",
    enable_prefix_caching=False,
    gpu_memory_utilization=0.85, # A30 has 24GB VRAM, can afford to use more
    max_model_len=16384,
    trust_remote_code=True,
    max_num_seqs=32,
    max_num_batched_tokens=131072,
)

sampling_params = SamplingParams(
    n=3,
    max_tokens=38912,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    logprobs=1,
)

print("Model loaded.")

# %% [markdown]
# ## 6. Generate Responses
# 
# We format every question into a chat-template prompt, then call `llm.generate()` in one batched pass.  
# vLLM handles batching and scheduling internally — no manual batching needed.

# %%
# Build prompts for first 10 entries
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

# Generate
print(f"Generating responses for {len(prompts)} questions...")
outputs = llm.generate(prompts, sampling_params=sampling_params)

def get_majority_voted_response(output):
    """
    Extract the boxed answer from each generation, find the majority vote,
    and return the full text of the first generation that produced that answer.
    This is a self-consistency method.
    """
    def clean_text(text):
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    candidates = []
    for o in output.outputs:
        cleaned = clean_text(o.text)
        boxed_answer = extract_last_boxed(cleaned)
        if boxed_answer is not None:
            candidates.append((boxed_answer, o.text.strip()))

    if not candidates:
        # Fallback to the highest logprob response if no boxed answers are found
        def score(o):
            if not o.logprobs: return float('-inf')
            total = sum(max(v.logprob for v in step.values()) for step in o.logprobs)
            return total
        best_by_prob = max(output.outputs, key=score)
        return best_by_prob.text.strip()

    answer_counts = Counter(ans for ans, txt in candidates)
    most_common_answer = answer_counts.most_common(1)[0][0]
    
    # Return the full text of the first response that gave the majority answer
    for ans, txt in candidates:
        if ans == most_common_answer:
            return txt
            
    return candidates[0][1]

# Using majority voting (self-consistency) instead of picking by log probability
responses = [get_majority_voted_response(out) for out in outputs]
    
# Preview first 3
for i in range(min(3, len(responses))):
    print(f"\n── Response {i} (id={data[i].get('id')}) ──")
    print(responses[i][:400], "..." if len(responses[i]) > 400 else "")


# %% [markdown]
# ## 7. Score Responses
# 
# Scoring differs by question type:
# 
# - **MCQ**: extract the predicted letter from `\boxed{}` and compare to the gold letter (exact match).
# - **Free-form**: use `Judger.auto_judge()` which handles symbolic and numeric equivalence.
# 
# Each result record contains `{id, is_mcq, gold, response, correct}`.

# %%
def strip_thinking(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
def latex_to_numeric(text: str) -> str:
    """Convert \\frac{a}{b} and \\dfrac{a}{b} to decimal strings"""
    def replace_frac(m):
        try:
            num, den = int(m.group(1)), int(m.group(2))
            return str(float(Fraction(num, den)))
        except:
            return m.group(0)
    text = re.sub(r'\\d?frac\{(\d+)\}\{(\d+)\}', replace_frac, text)
    return text
    
def extract_letter(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    matches = re.findall(r'\\boxed\{([A-Za-z])\}', text)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r'\b([A-Z])\b', text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


# Load Judger for free-form scoring
sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

#results = []
### for public test comment out from here
# for item, response in tqdm(zip(data, responses), total=len(responses), desc="Scoring"): #chnge data[:20] or [-100:] to check depends on 
#     is_mcq = bool(item.get("options"))
#     gold   = item["answer"]
#     raw_response = response
#     response = strip_thinking(response)
    
#     if is_mcq:
#         correct = score_mcq(response, str(gold))
#     else:
#         gold_list = gold if isinstance(gold, list) else [gold]
#         last_boxed = extract_last_boxed(response)
#         pred_converted = latex_to_numeric(last_boxed) if last_boxed is not None else ""
#         pred = f"\\\\boxed{{{pred_converted}}}" if pred_converted else response
#         try:
#             correct = judger.auto_judge(
#                 pred=pred,
#                 gold=gold_list,
#                 options=[[]] * len(gold_list),
#             )
#         except Exception:
#             correct = False

#     results.append({
#         "id":       item.get("id"),
#         "is_mcq":   is_mcq,
#         "gold":     gold,
#         "response": raw_response,
#         "correct":  correct,
#     })

# print(f"Scoring complete. {len(results)} results.")


#### for private set comment out
results = []
for item, response in zip(data, responses):
    results.append({
        "id":       item.get("id"),
        "response": response,  # raw full response for submission
    })

print(f"Collected {len(results)} results")


# %%
for r in results[:10]:
    predicted = extract_last_boxed(r["response"])
    print(f"\nid={r['id']}")
    #print(f"  Gold:      {r['gold']}")
    #print(f"  Predicted: {predicted}")
    #print(f"  Correct:   {r['correct']}")

# %% [markdown]
# ## 8. Summary
# 
# Print accuracy broken down by question type.

# %%
# mcq_res  = [r for r in results if r["is_mcq"]]
# free_res = [r for r in results if not r["is_mcq"]]

# def acc(subset):
#     return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

# print("=" * 50)
# print("EVALUATION RESULTS")
# print("=" * 50)
# print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
# print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
# print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
# print("=" * 50)

# %% [markdown]
# ## 9. Save Results
# 
# Results are written as newline-delimited JSON.
# 
# **With evaluation** (public set — you have ground-truth):  
# Each line: `{id, is_mcq, gold, response, correct}`
# 
# **Without evaluation** (private test set — no ground-truth available):  
# Each line: `{id, is_mcq, response}` — omit `gold` and `correct`.
# 
# Toggle `SAVE_EVAL` below accordingly.

# %%
SAVE_EVAL = False   # Set to False when running on the private test set

out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    for r in results:
        if SAVE_EVAL:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                      "response": r["response"], "correct": r["correct"]}
        else:
            record = {"id": r["id"], "response": r["response"]}
        f.write(json.dumps(record) + "\n")

print(f"Saved {len(results)} records to {out_path}")

# %%
# Save submission CSV
import csv
submission_path = Path("results/submission.csv")

with open(submission_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["id", "response"])
    for r in results:
        writer.writerow([r["id"], r["response"]])

print(f"Saved submission CSV to {submission_path}")

# Sanity check
import pandas as pd
df = pd.read_csv(submission_path)
print(f"Shape: {df.shape}")
print(df.head(3))


