#!/usr/bin/env python
# coding: utf-8

# # Derived from https://www.philschmid.de/gptq-llama
# 
# # Quantize open LLMs using optimum and GPTQ
# 
# The Hugging Face Optimum team collaborated with AutoGPTQ library to provide a simple API that apply GPTQ quantization on language models. With GPTQ quantization open LLMs to 8, 4, 3 or even 2 bits to run them on smaller Hardware without a big drop of performance. 
# 
# In the blog, you will learn how to:
# 
# 1. Setup our development environment
# 2. Prepare quantization dataset
# 3. Load and Quantize Model
# 4. Test performance and inference speed
# 5. Bonus: Run Inference with Text Generation Inference
#   
# But we before we get started lets take quick look on what GPTQ does. 
# 
# _Note: This tutorial was created and run on a g5.2xlarge AWS EC2 Instance, including an NVIDIA A10G GPU._
# 
# 
# ## What is GPTQ?
# 
# [GPTQ](https://arxiv.org/abs/2210.17323) is a post-training quantziation method to compress LLMs, like GPT. GPTQ compresses GPT models by reducing the number of bits needed to store each weight in the model, from 32 bits down to just 3-4 bits. This means the model takes up much less memory, so it can run on less Hardware, e.g. Single GPU for 13B Llama2 models. GPTQ analyzes each layer of the model separately and approximating the weights in a way that preserves the overall accuracy.
# 
# The main benefits are:
# * Quantizes the weights of the model layer-by-layer to 4 bits instead of 16 bits, this reduces the needed memory by 4x.
# * Quantization is done gradually to minimize the accuracy loss from quantization.
# * Achieves same latency as fp16 model, but 4x less memory usage, sometimes faster due to custom kernels, e.g. [Exllama](https://github.com/turboderp/exllama)
# * Quantized weights can be saved to disk for a head of time quantization.
# 
# _Note: GPTQ quantization only works for text model for now. Futhermore, the quantization process can take a lot of time. You check on the [Hugging Face Hub](https://huggingface.co/models?search=gptq) if there is not already a GPTQ quantized version of the model you want to use._
# 
# --- 
# 
# ## 1. Setup our development environment
# 
# Let's start coding, but first, install our dependencies.

# In[2]:


# !pip install "torch==2.0.1" "transformers==4.32.1" "optimum==1.12.0" "auto-gptq==0.4.2" "accelerate==0.22.0" "safetensors>=0.3.1" --upgrade

# ## 2. Prepare quantization dataset
# 
# GPTQ is a post-training quantization method, so we need to prepare a dataset to quantize our model. We can either use a dataset from the [Hugging Face Hub](https://huggingface.co/datasets) or use our own dataset. In this blog, we are going to use the [WikiText](https://huggingface.co/datasets/wikitext) dataset from the Hugging Face Hub. The dataset is used to quantize the weights to minimize the performance loss. It is recommended to use a quantization dataset with atleast `128` samples.
# 
# _Note: [TheBloke](https://huggingface.co/TheBloke) a very active community member is contributing hundreds of gptq weights to the Hugging Face Hub. He mostly uses wikitext as quantization dataset for general domain models._
# 
# If you want to use, e.g. your fine-tuning dataset for quantization you can provide it as a list instead of the "id", check out this [example](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb).  

# In[3]:


# Dataset id from Hugging Face 

dataset_id = "wikitext2"

# ## 3. Load and Quantize Model
# 
# Optimum integrates GPTQ quantization in the `optimum.qptq` namespace with a `GPTQQuantizer`. The quantizer takes our dataset (id or list), bits, and model_seqlen as input. For more customization check [here](https://github.com/huggingface/optimum/blob/234a427450a7dcc978b227fa627ebcdab1764318/optimum/gptq/quantizer.py#L76).
# 

# In[4]:


from optimum.gptq import GPTQQuantizer

# GPTQ quantizer
quantizer = GPTQQuantizer(bits=4, dataset=dataset_id, model_seqlen=4096)
quantizer.quant_method = "gptq"

# After we have created our Quantizer we can load our model using Transformers. In our example we will quantize a [Llama 2 7B](https://huggingface.co/philschmid/llama-2-7b-instruction-generator), which we trained in my other blog post ["Extended Guide: Instruction-tune Llama 2"](https://www.philschmid.de/instruction-tune-llama-2). We are going to load our model in `fp16` since GPTQ adopts a mixed int4/fp16 quantization scheme where weights are quantized as int4 while activations remain in float16. 

# In[5]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hugging Face model id
model_checkpoint = "NousResearch/Llama-2-7b-hf" # non-gated
#model_checkpoint = "meta/llama-2-7b" # gated

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False) # bug with fast tokenizer
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.float16) # we load the model in fp16 on purpose

# After we loaded our model we are ready to quantize it. 
# _Note: Quantization can take process can take a lot of time depending on one's hardware. For this example the quantization on a single A10G GPU for a 7B model took ~minutes._ 

# In[6]:


import os 
import json

# quantize the model 
quantized_model = quantizer.quantize_model(model, tokenizer)

# save the quantize model to disk
save_folder = "quantized_llama"
model.save_pretrained(save_folder, safe_serialization=True)

# load fresh, fast tokenizer and save it to disk
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint).save_pretrained(save_folder)

# save quantize_config.json for TGI 
with open(os.path.join(save_folder, "quantize_config.json"), "w", encoding="utf-8") as f:
  quantizer.disable_exllama = False
  json.dump(quantizer.to_dict(), f, indent=2)

# since the model was partially offloaded it set `disable_exllama` to `True` to avoid an error. For inference and production load we want to leverage the exllama kernels. Therefore we need to change the `config.json`

# In[7]:


with open(os.path.join(save_folder, "config.json"), "r", encoding="utf-8") as f:
  config = json.load(f)
  config["quantization_config"]["disable_exllama"] = False
  with open(os.path.join(save_folder, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

# ## 4. Test performance and inference speed
# 
# Since the latest release of transformers we can load any GPTQ quantized model directly using the `AutoModelForCausalLM` class this. You can either load already quantized models from Hugging Face, e.g. [TheBloke/Llama-2-13B-chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ) or models you quantized yourself. Since we want to test here the results of our quantization we are going to load our quantized model from disk and compare it to our non quantize model. 
# 
# First lets our our non quantized model and test it on a simple prompt.

# In[8]:


import time 

# The prompt is based on the fine-tuning from the model: https://www.philschmid.de/instruction-tune-llama-2#4-test-model-and-run-inference
# prompt = """### Instruction:
# Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

# ### Input:
# Dear [boss name],

# I'm writing to request next week, August 1st through August 4th,
# off as paid time off.

# I have some personal matters to attend to that week that require
# me to be out of the office. I wanted to give you as much advance
# notice as possible so you can plan accordingly while I am away.

# Thank you, [Your name]

# ### Response:
# """

prompt = """### Instruction:
What are some common ways to deploy a model on AWS?

### Response:
"""


# helper function to generate text and measure latency
def generate_helper(pipeline,prompt=prompt):
    # warm up
    for i in range(5):
      _ = pipeline("Warm up")

    # measure latency in a simple way 
    start = time.time()
    out = pipeline(prompt, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)
    end = time.time()
    
    generated_text = out[0]["generated_text"][len(prompt):]
    
    latency_per_token_in_ms = ((end-start)/len(pipeline.tokenizer(generated_text)["input_ids"]))*1000
    
    # return the generated text and the latency
    return {"text": out[0]["generated_text"][len(prompt):], "latency": f"{round(latency_per_token_in_ms,2)}ms/token"}


# We can load the vanilla transformers model and run inference using the `pipeline` class. 

# In[9]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Hugging Face model id
model_checkpoint = "NousResearch/Llama-2-7b-hf" # non-gated
#model_checkpoint = "meta/llama-2-7b" # gated

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto", torch_dtype=torch.float16) # we load the model in fp16 on purpose

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Create the base line

# In[10]:


import torch 

vanilla_res = generate_helper(pipe)

print(f"Latency: {vanilla_res['latency']}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Generated Instruction: {vanilla_res['text']}")

# Latency: 37.49ms/token
# GPU memory: 12.62 GB
# Generated Instruction: Write a request for PTO letter to my boss

# In[11]:


# clean up 
del pipe
del model 
del tokenizer
torch.cuda.empty_cache()

# Since we have now our baseline we can test and validate our GPTQ quantize weights. Therefore we will use the new `gptq` integration into the `AutoModelForCausalLM` class where we can directly load the `gptq` weights. 

# In[12]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# path to gptq weights
quantized_model_checkpoint = "quantized_llama"

q_tokenizer = AutoTokenizer.from_pretrained(quantized_model_checkpoint)
q_model = AutoModelForCausalLM.from_pretrained(quantized_model_checkpoint, device_map="auto", torch_dtype=torch.float16)

qtq_pipe = pipeline("text-generation", model=q_model, tokenizer=q_tokenizer)

# Now, we can test our quantized model on the same prompt as our baseline.

# In[13]:


gpq_res = generate_helper(qtq_pipe)

print(f"Latency: {gpq_res['latency']}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Generated Instruction: {gpq_res['text']}")

# Latency: 36.0ms/token
# GPU memory: 3.83 GB
# Generated Instruction: Write a letter requesting time off

# For comparison the vanilla model needed ~12.6GB Memory and the GPTQ model needed ~3.8GB Memory, with equal performance. GPTQ allowed us to save ~4x memory (don't forget pytorch has default kernels). 

# With Text Generation inference we are achieving ~`22.942983ms` latency per token, which is 2x faster than transformers. If you plan to deploy your model in production, I would recommend to use Text Generation Inference.

# 
