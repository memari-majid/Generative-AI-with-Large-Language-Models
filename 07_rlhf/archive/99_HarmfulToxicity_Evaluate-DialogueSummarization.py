#!/usr/bin/env python
# coding: utf-8

# In[6]:


%pip install --disable-pip-version-check torch==1.13.1 torchdata==0.5.1

# In[9]:


%pip install --disable-pip-version-check -q \
    transformers==4.27.2 \
    datasets==2.9.0 \
    accelerate==0.17.0 \
    evaluate==0.4.0 \
    trl==0.4.1

# In[37]:


%pip install git+https://github.com/huggingface/peft.git

# In[45]:


import argparse
import csv

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="test")

model_type = "all"
output_file = "toxicity.csv"
batch_size=64
num_samples=400
context_length=2000
max_new_tokens=30


if model_type == "all":
    MODELS_TO_TEST = [
        "google/flan-t5-base",
        "./instruct-dialogue-summary-checkpoint/",
        "./ppo-dialogue-summary-checkpoint/",
        "./peft-dialogue-summary-checkpoint/",
    ]
    
NUM_SAMPLES = num_samples
BATCH_SIZE = batch_size
output_file = output_file
max_new_tokens = max_new_tokens
context_length = context_length
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

# consider only toxic prompts
ds = ds.filter(lambda x: x["label"] == 1)

toxicities = {}

# open a csv file
file = open(f"{output_file}", "w", newline="")
writer = csv.writer(file)
# add first rows
writer.writerow(["model_id", "mean_toxicity", "std_toxicity"])

from peft import PeftModel

for model_id in tqdm(MODELS_TO_TEST):
    print(model_id)
    if 'peft' in model_id:
        peft_model_base = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base', device_map={"": device}, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(peft_model_base, model_id, device_map={"": device}, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    input_texts = []

    for i, example in enumerate(ds):
        # set seed
        torch.manual_seed(42)

        input_text = example["comment_text"]
        input_texts.append(input_text[:2000])

        if i > NUM_SAMPLES:
            break

        if (i + 1) % BATCH_SIZE == 0:
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
            inputs.input_ids = inputs.input_ids[:context_length]
            inputs.attention_mask = inputs.attention_mask[:context_length]
            outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [
                generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
            ]
            toxicity_score = toxicity.compute(predictions=generated_texts)
            input_texts = []

            if model_id not in toxicities:
                toxicities[model_id] = []
            toxicities[model_id].extend(toxicity_score["toxicity"])

    # last batch
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=30)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_texts = [generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)]
    toxicity_score = toxicity.compute(predictions=generated_texts)
    toxicities[model_id].extend(toxicity_score["toxicity"])

    # compute mean & std using np
    mean = np.mean(toxicities[model_id])
    std = np.std(toxicities[model_id])

    # save to file
    writer.writerow([model_id, mean, std])

    # print
    print(f"Model: {model_id} - Mean: {mean} - Std: {std}")

    model = None
    torch.cuda.empty_cache()

# close file
file.close()

# In[42]:


toxicity_metrics

# In[ ]:


print("""
flan-t5-base
  Mean: 0.10735442655624947
  Std: 0.2648808616377917

flan-t5-instruct-full
  Mean: 0.12531453278060511
  Std: 0.28167328436940203

flan-t5-instruct-peft
  Mean: 0.15925555246049858
  Std: 0.3149103276023309

flan-t5-instruct-peft-rl-detoxify
  Mean: 0.1088853306687247
  Std: 0.2653053856564228
""")

