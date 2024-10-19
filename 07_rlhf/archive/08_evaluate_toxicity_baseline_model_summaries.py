#!/usr/bin/env python
# coding: utf-8

# # REQUIRES LARGE GPU INSTANCE TYPE

# In[11]:


import psutil

notebook_memory = psutil.virtual_memory()
print(notebook_memory)

if notebook_memory.total < 32 * 1000 * 1000 * 1000:
    print('*******************************************')    
    print('YOU ARE NOT USING THE CORRECT INSTANCE TYPE')
    print('PLEASE CHANGE INSTANCE TYPE TO  m5.2xlarge ')
    print('*******************************************')
else:
    correct_instance_type=True

# # Quantitative Results with ROGUE Metric
# 
# The [ROUGE metric](https://en.wikipedia.org/wiki/ROUGE_(metric)) helps quantify the validity of summarizations produced by models. It compares summarizations to a "baseline" summary which is usually created by a human. While not perfect, it does give an indication to the overall increase in summarization effectiveness that we have accomplished by fine-tuning.

# In[12]:


%pip install --disable-pip-version-check -q \
    transformers==4.27.2 \
    datasets==2.9.0 \
    accelerate==0.17.0 \
    promptsource==0.2.3 \
    evaluate==0.4.0 \
    trl==0.4.1 \
    rouge_score==0.1.2 \
    loralib==0.1.1

# In[ ]:


import argparse
import csv

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

toxicity_evaluator = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="test")

# parser = argparse.ArgumentParser(description="Evaluate de-toxified models")
# parser.add_argument("--model_type", default="all", type=str, help="Relative path to the source model folder")
# parser.add_argument("--output_file", default="toxicity.csv", type=str, help="Relative path to the source model folder")
# parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
# parser.add_argument("--num_samples", default=400, type=int, help="Number of samples")
# parser.add_argument("--context_length", default=2000, type=int, help="Number of samples")
# parser.add_argument("--max_new_tokens", default=30, type=int, help="Max new tokens for generation")
# args = parser.parse_args()

class Args:
    model_type: str
    output_file: str
    batch_size: float
    num_samples: float
    content_length: float
    max_new_tokens: str

args = Args()    
args.model_type='all'
args.output_file='toxicity.csv'
args.batch_size=64
args.num_samples=400
args.context_length=2000
args.max_new_tokens=30


if args.model_type == "all":
    MODELS_TO_TEST = [
        "t5-base",
        "google/flan-t5-base",
    ]
    
NUM_SAMPLES = args.num_samples
BATCH_SIZE = args.batch_size
output_file = args.output_file
max_new_tokens = args.max_new_tokens
context_length = args.context_length
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
# consider only toxic prompts
ds = ds.filter(lambda x: x["label"] == 1)

toxicities = {}

# open a csv file
file = open(f"{output_file}", "w", newline="")
writer = csv.writer(file)
# add first rows
writer.writerow(["model_id", "mean_toxicity", "std_toxicity"])


for model_id in tqdm(MODELS_TO_TEST):
    # load base LLM model and tokenizer
    if 'peft' in model_id:
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_checkpoint, device_map={"": device}, torch_dtype=torch.bfloat16)  #, device_map="auto")

        # Load the LoRA/PEFT model
        model = PeftModel.from_pretrained(pretrained_model, model_id, device_map={"": device}, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
            toxicity_score = toxicity_evaluator.compute(predictions=generated_texts)
            input_texts = []

            if model_id not in toxicities:
                toxicities[model_id] = []
            toxicities[model_id].extend(toxicity_score["toxicity"])

    # last batch
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=30)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_texts = [generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)]
    toxicity_score = toxicity_evaluator.compute(predictions=generated_texts)
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

# # Release Resources

# In[ ]:


%%html

<p><b>Shutting down your kernel for this notebook to release resources.</b></p>
<button class="sm-command-button" data-commandlinker-command="kernelmenu:shutdown" style="display:none;">Shutdown Kernel</button>
        
<script>
try {
    els = document.getElementsByClassName("sm-command-button");
    els[0].click();
}
catch(err) {
    // NoOp
}    
</script>

# In[ ]:



