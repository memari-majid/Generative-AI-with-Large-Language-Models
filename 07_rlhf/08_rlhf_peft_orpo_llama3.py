#!/usr/bin/env python
# coding: utf-8

# # Fine-tune Llama 3 with ORPO
# 
# Based on the following notebook:  [Large Language Model Course](https://github.com/mlabonne/llm-course)
# 
# Uses this dataset:  https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k
# 
# Note:  This requires a high-memory GPU such as the p4d, p4de, or p5+ instance types.
# 
# ![](img/from_rlhf_to_dpo_to_orpo.png)

# In[ ]:


# %pip install -U torch torchvision transformers==4.39.0 datasets peft trl accelerate bitsandbytes
# #%pip install flash-attn --no-build-isolation


# In[ ]:


import gc
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

# Model
#base_model = "meta-llama/Meta-Llama-3-8B"
base_model = "NousResearch/Meta-Llama-3-8B"
new_model = "OrpoLlama-3-8B"

# Set torch dtype and attention implementation
# if torch.cuda.get_device_capability()[0] >= 8:
#     !pip install -qqq flash-attn
#     torch_dtype = torch.bfloat16
#     attn_implementation = "flash_attention_2"
# else:
torch_dtype = torch.float16
attn_implementation = "eager"

# In[ ]:


# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: " + device)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map=device, #"auto",
    attn_implementation=attn_implementation
)
model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

# In[ ]:


dataset_name = "mlabonne/orpo-dpo-mix-40k"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=42).select(range(1000)) # Only use 1000 samples for quick demo

def format_chat_template(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc= os.cpu_count(),
)
dataset = dataset.train_test_split(test_size=0.01)

# In[ ]:


orpo_args = ORPOConfig(
    learning_rate=8e-6,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    beta=0.1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    #optim="adamw_torch",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    output_dir="./results/",
    report_to="none"
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(new_model)

# In[ ]:


# Flush memory
del trainer, model
gc.collect()
gc.collect()
torch.cuda.empty_cache()

# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)
fp16_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
fp16_model, tokenizer = setup_chat_format(fp16_model, tokenizer)

# Merge adapter with base model
model = PeftModel.from_pretrained(fp16_model, new_model)
model = model.merge_and_unload()

# In[ ]:



