#!/usr/bin/env python
# coding: utf-8

# # Tested on ml.p4de.24xlarge
# PyTorch 2.0.0, Python 3.10, CUDA Version 11.8

# In[2]:


# !nvidia-smi

# In[4]:


# Matches the CUDA version reported by nvidia-smi (11.8, in this case)
# Configure here: https://pytorch.org/
# %pip install -U torch --index-url https://download.pytorch.org/whl/cu118
# %pip install -U transformers==4.36.2
# %pip install -U peft==0.7.1
# %pip install -U datasets==2.15.0
# %pip install -U bitsandbytes==0.41.2
# %pip install -U scipy==1.10.1
# %pip install -U ipywidgets==8.1.1
# %pip install -U matplotlib==3.7.4

# In[6]:


# !python -m bitsandbytes

# In[2]:


from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# In[3]:


from datasets import load_dataset

dataset_name = "databricks/databricks-dolly-15k"

train_dataset = load_dataset(dataset_name, split="train[0:800]")
eval_dataset = load_dataset(dataset_name, split="train[800:1000]")

# In[4]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, 
                                             quantization_config=bnb_config, 
                                             device_map="auto")

# ## Setup instruction dataset
# 
# Must follow this format:  https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

# In[41]:


# Tokenization 
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# In[42]:


def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""

[INST] Given a question and some additional context, provide an answer. [/INST]

### Question:
{data_point['instruction']}

### Context:
{f"Here is some context: {data_point['context']}" if len(data_point["context"]) > 0 else ""}

### Response:
{data_point['response']}

</s>
"""

    tokenized_prompt = tokenizer(full_prompt)
    return tokenized_prompt

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

untokenized_text = tokenizer.decode(tokenized_train_dataset[0]['input_ids']) 
print(untokenized_text)

# In[19]:


# #max_length = 400 

# tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
# tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

# untokenized_text = tokenizer.decode(tokenized_train_dataset[4]['input_ids']) 
# print(untokenized_text)

# In[20]:


from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# In[21]:


from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "w1",
        "w2",
        "w3",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)

# In[22]:


!rm -rf ./dolly_mixtral_finetune

# In[23]:


import transformers
from datetime import datetime

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

output_dir = "./dolly_mixtral_finetune"

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=10,
        learning_rate=2.5e-5, 
        logging_steps=5,
        fp16=True, 
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=10,               # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=10,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# In[39]:


eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left"
)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

# In[46]:


prompt = """
<s>
[INST] Given a question and some additional context, provide an answer. [/INST]

### Question:
Who wrote "Generative AI on AWS" by O'Reilly Media?

### Context: 
Chris Fregly and Antje Barth wrote "Data Science on AWS" by O'Reilly Media.
Chris Fregly, Antje Barth, and Shelbee Eigenbrode wrote "Generative AI on AWS" by O'Reilly Media.

### Response:</s>
"""

model_input = eval_tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=32)[0], skip_special_tokens=True))

# # Load PEFT model and perform inference

# In[24]:


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# base_model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_id, 
#     quantization_config=bnb_config,
#     device_map="auto"
# )

# In[ ]:


# from peft import PeftModel

# peft_model = PeftModel.from_pretrained(base_model, f"{output_dir}/checkpoint-10")
# #peft_model.eval()

# In[ ]:


# tokenizer = AutoTokenizer.from_pretrained(
#     base_model_id,
# #    add_bos_token=True
# )
# tokenizer.pad_token = tokenizer.eos_token

# eval_tokenizer = AutoTokenizer.from_pretrained(
#     base_model_id,
#     add_bos_token=True,
# )

# In[ ]:


# prompt = """

# [INST] Given a question and some additional context, provide an answer. [/INST]

# ### Question:
# Who wrote "Generative AI on AWS" by O'Reilly Media?

# ### Context: 
# Chris Fregly and Antje Barth wrote "Data Science on AWS" by O'Reilly Media.
# Chris Fregly, Antje Barth, and Shelbee Eigenbrode wrote "Generative AI on AWS" by O'Reilly Media.

# ### Response:

# """

# model_input = tokenizer(prompt, return_tensors="pt").to("cuda")

# with torch.no_grad():
#     print(tokenizer.decode(peft_model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

# In[ ]:



