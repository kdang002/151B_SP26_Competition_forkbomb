import json

with open("151b_model_google_colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell indices
data_load_idx = -1
prompt_const_idx = -1
vllm_load_idx = -1

for i, cell in enumerate(nb["cells"]):
    if "data = [json.loads" in "".join(cell.get("source", [""])):
        data_load_idx = i
    elif "SYSTEM_PROMPT_MATH =" in "".join(cell.get("source", [""])):
        prompt_const_idx = i
    elif "tokenizer = AutoTokenizer.from_pretrained(" in "".join(cell.get("source", [""])):
        vllm_load_idx = i

print(f"data_load_idx: {data_load_idx}, prompt_const_idx: {prompt_const_idx}, vllm_load_idx: {vllm_load_idx}")

split_cell = {
 "cell_type": "code",
 "execution_count": None,
 "id": "split_data_cell",
 "metadata": {},
 "outputs": [],
 "source": [
  "import random\n",
  "from sklearn.model_selection import train_test_split\n",
  "\n",
  "# Set random seed for reproducibility\n",
  "random.seed(42)\n",
  "\n",
  "# Create train/val/test splits (80 / 10 / 10)\n",
  "train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)\n",
  "val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)\n",
  "\n",
  "print(f\"Train size: {len(train_data)}\")\n",
  "print(f\"Val size:   {len(val_data)}\")\n",
  "print(f\"Test size:  {len(test_data)}\")\n"
 ]
}

format_cell = {
 "cell_type": "code",
 "execution_count": None,
 "id": "format_sft_cell",
 "metadata": {},
 "outputs": [],
 "source": [
  "from transformers import AutoTokenizer\n",
  "from datasets import Dataset\n",
  "\n",
  "tokenizer_sft = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)\n",
  "tokenizer_sft.pad_token = tokenizer_sft.eos_token\n",
  "\n",
  "def format_sft_dataset(dataset_split):\n",
  "    formatted = []\n",
  "    for item in dataset_split:\n",
  "        system, user = build_prompt(item[\"question\"], item.get(\"options\"))\n",
  "        ans = item[\"answer\"]\n",
  "        if isinstance(ans, list):\n",
  "            ans = ans[0]\n",
  "        messages = [\n",
  "            {\"role\": \"system\", \"content\": system},\n",
  "            {\"role\": \"user\", \"content\": user},\n",
  "            {\"role\": \"assistant\", \"content\": f\"The final answer is \\\\boxed{{{ans}}}.\"}\n",
  "        ]\n",
  "        text = tokenizer_sft.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)\n",
  "        formatted.append({\"text\": text})\n",
  "    return Dataset.from_list(formatted)\n",
  "\n",
  "train_dataset = format_sft_dataset(train_data)\n",
  "val_dataset = format_sft_dataset(val_data)\n",
  "print(f\"Formatted train samples: {len(train_dataset)}\")\n"
 ]
}

train_cell = {
 "cell_type": "code",
 "execution_count": None,
 "id": "train_sft_cell",
 "metadata": {},
 "outputs": [],
 "source": [
  "import torch\n",
  "import gc\n",
  "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
  "from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments\n",
  "from trl import SFTTrainer\n",
  "from transformers import EarlyStoppingCallback\n",
  "\n",
  "# Clear any leftover memory\n",
  "torch.cuda.empty_cache()\n",
  "gc.collect()\n",
  "\n",
  "# Quantization Config for QLoRA\n",
  "bnb_config = BitsAndBytesConfig(\n",
  "    load_in_4bit=True,\n",
  "    bnb_4bit_use_double_quant=True,\n",
  "    bnb_4bit_quant_type=\"nf4\",\n",
  "    bnb_4bit_compute_dtype=torch.float16\n",
  ")\n",
  "\n",
  "print(\"Loading base model for training...\")\n",
  "model = AutoModelForCausalLM.from_pretrained(\n",
  "    MODEL_ID,\n",
  "    quantization_config=bnb_config,\n",
  "    device_map={\"\" : 0},\n",
  "    trust_remote_code=True\n",
  ")\n",
  "model = prepare_model_for_kbit_training(model)\n",
  "\n",
  "# LoRA Config\n",
  "lora_config = LoraConfig(\n",
  "    r=16,\n",
  "    lora_alpha=32,\n",
  "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
  "    lora_dropout=0.05,\n",
  "    bias=\"none\",\n",
  "    task_type=\"CAUSAL_LM\"\n",
  ")\n",
  "model = get_peft_model(model, lora_config)\n",
  "model.print_trainable_parameters()\n",
  "\n",
  "training_args = TrainingArguments(\n",
  "    output_dir=\"./results/qwen_lora\",\n",
  "    per_device_train_batch_size=2,\n",
  "    gradient_accumulation_steps=4,\n",
  "    learning_rate=2e-5,\n",
  "    lr_scheduler_type=\"cosine\",\n",
  "    warmup_ratio=0.1,\n",
  "    num_train_epochs=1,\n",
  "    logging_steps=10,\n",
  "    eval_strategy=\"steps\",\n",
  "    eval_steps=50,\n",
  "    save_strategy=\"steps\",\n",
  "    save_steps=50,\n",
  "    fp16=True,\n",
  "    report_to=\"none\",\n",
  "    optim=\"paged_adamw_8bit\",\n",
  "    load_best_model_at_end=True,\n",
  "    metric_for_best_model=\"eval_loss\"\n",
  ")\n",
  "\n",
  "trainer = SFTTrainer(\n",
  "    model=model,\n",
  "    train_dataset=train_dataset,\n",
  "    eval_dataset=val_dataset,\n",
  "    peft_config=lora_config,\n",
  "    dataset_text_field=\"text\",\n",
  "    max_seq_length=1024,\n",
  "    tokenizer=tokenizer_sft,\n",
  "    args=training_args,\n",
  "    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]\n",
  ")\n",
  "\n",
  "print(\"Starting training...\")\n",
  "trainer.train()\n",
  "\n",
  "new_model_path = \"./qwen_lora_final\"\n",
  "trainer.model.save_pretrained(new_model_path)\n",
  "tokenizer_sft.save_pretrained(new_model_path)\n",
  "print(f\"LoRA adapter saved to {new_model_path}\")\n",
  "\n",
  "# Clear memory before loading vLLM\n",
  "del model\n",
  "del trainer\n",
  "torch.cuda.empty_cache()\n",
  "gc.collect()\n",
  "print(\"GPU memory cleared. Ready for vLLM inference.\")\n"
 ]
}

cells = nb["cells"]
# Insert split cell after data load
cells.insert(data_load_idx + 1, split_cell)
# Note: indices shift
prompt_const_idx += 1
vllm_load_idx += 1

# Insert format cell after prompt construction
cells.insert(prompt_const_idx + 1, format_cell)
vllm_load_idx += 1

# Insert training cell before vLLM load
cells.insert(vllm_load_idx, train_cell)
vllm_load_idx += 1

# Update vLLM load cell to load LoRA
vllm_cell = cells[vllm_load_idx]
source = "".join(vllm_cell["source"])
source = source.replace("enable_prefix_caching=False,", "enable_prefix_caching=False,\n    enable_lora=True,\n    max_lora_rank=16,")
vllm_cell["source"] = [s + "\n" for s in source.split("\n")]
vllm_cell["source"][-1] = vllm_cell["source"][-1].rstrip("\n")

# Update generation cell to use lora_request
for cell in cells:
    if "llm.generate(" in "".join(cell.get("source", [""])):
        gen_source = "".join(cell["source"])
        if "from vllm.lora.request import LoRARequest" not in gen_source:
            gen_source = gen_source.replace("with torch.no_grad():", "from vllm.lora.request import LoRARequest\n    with torch.no_grad():")
            # Replace "do_sample=True,\n        )" with the LoRARequest
            import re
            gen_source = re.sub(r"(do_sample=True,.*?)(?=\n\s*\))", r"\1\n            lora_request=LoRARequest(\"qwen_adapter\", 1, \"./qwen_lora_final\")", gen_source, flags=re.DOTALL)
            cell["source"] = [s + "\n" for s in gen_source.split("\n")]
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

nb["cells"] = cells

with open("151b_model_google_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")