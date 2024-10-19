#!/usr/bin/env python
# coding: utf-8

# In[13]:


%pip install torch==2.0.0 torchdata

%pip install --disable-pip-version-check -q \
    transformers==4.27.2 \
    datasets==2.9.0 \
    accelerate==0.17.0 \
    evaluate==0.4.0 \
    trl==0.4.1 \
    rouge_score==0.1.2 \
    loralib==0.1.1

# In[14]:


!pip install git+https://github.com/huggingface/peft.git

# In[15]:


!pip install git+https://github.com/lvwerra/trl.git

# In[25]:


from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

tqdm.pandas()

#model_name="lvwerra/t5-imdb"
model_name="google/flan-t5-base"
log_with=None
learning_rate=5e-5
mini_batch_size=16
batch_size=256
gradient_accumulation_steps=1

# In[44]:


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.

config = PPOConfig(
    model_name=model_name,
    learning_rate=learning_rate,
    log_with=log_with,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
)
# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
# def build_imdb_dataset(tokenizer, input_min_text_length=2, input_max_text_length=8):
#     # load imdb with datasets
#     ds = load_dataset("imdb", split="train")
#     ds = ds.rename_columns({"text": "review"})
#     ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

#     input_size = LengthSampler(input_min_text_length, input_max_text_length)

#     def tokenize(sample):
#         sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()] + [tokenizer.eos_token_id]
#         sample["query"] = tokenizer.decode(sample["input_ids"])
#         return sample

#     ds = ds.map(tokenize, batched=False)
#     ds.set_format(type="torch")
#     return ds


def build_dialogsum_dataset(tokenizer, input_min_text_length=2, input_max_text_length=8):
    # load dialogsum with datasets (columns:  dialog, summary)
    ds = load_dataset("knkarthick/dialogsum", split="train")
#    ds = ds.rename_columns({"summary": "review"})
    ds = ds.filter(lambda x: len(x["summary"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["summary"])[: input_size()] + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dialogsum_dataset(tokenizer)

#query = tokenizer("I really liked this movie because", return_tensors="pt")["input_ids"]

generation_kwargs = {"top_k": 0.0, "top_p": 1.0, "do_sample": True, "eos_token_id": -1}


# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collater)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
sentiment_pipe = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=device)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
output_min_length = 16
output_max_length = 32
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
        
    # Get response from t5
    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, length_sampler=output_length_sampler, **generation_kwargs
    )
    response_tensors = [r[1:] for r in response_tensors]
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]).to(device) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

# In[45]:


model_save_path = './ppo-dialogue-summary-checkpoint/'

#ppo_trainer.save_pretrained(model_save_path) # this relies on huggingface_hub 
#ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained(model_save_path) # not sure what this does
ppo_trainer.model.save_pretrained(model_save_path)
ppo_trainer.tokenizer.save_pretrained(model_save_path)

# In[46]:


#text_generation_pipeline = pipeline('text-generation', './ppo-dialogue-summary-checkpoint/', device=device)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

ppo_model = AutoModelForSeq2SeqLM.from_pretrained(model_save_path)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)

# In[47]:


from datasets import load_dataset
dataset = load_dataset("knkarthick/dialogsum")
dataset

# # Test the Model with Zero-Shot Prompts Before Tuning
# 
# In the example below, we highlight how the summarization capability of the model is lacking compared to the baseline summary provided in the dataset. You can see that the model struggles to summarize the dialogue compared to the baseline summary, but it does pull out some important information from the text which indicates the model can be fine tuned to the task at hand.

# In[48]:


idx = 10
diag = dataset['test'][idx]['dialogue']
baseline_human_summary = dataset['test'][idx]['summary']

prompt = f'Summarize the following conversation.\n\n{diag}\n\nSummary:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

ppo_outputs = ppo_model.generate(input_ids, GenerationConfig(max_new_tokens=200))
ppo_text_output = tokenizer.decode(ppo_outputs[0], skip_special_tokens=True)

print(f'Prompt:\n--------------------------\n{prompt}\n--------------------------')
print(f'Baseline human summary : {baseline_human_summary}')
print(f'PPO summary: {ppo_text_output}')

# In[49]:


original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# again, for the sake of time, we will only be generating 10 summarizations with each model
# outside of the lab, a good exercise is to increase the number of validation summaries generated
dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']

original_model_summaries = []
# instruct_model_summaries = []
# peft_model_summaries = []
ppo_model_summaries = []

for idx, diag in enumerate(dialogues):
    prompt = f'Summarize the following conversation.\n\nConversation:\n{diag}\n\nSummary:'
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # human baseline
    human_baseline_text_output = human_baseline_summaries[idx]
    
    # original model
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    # instruct model
    # instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    # instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

    # peft model
    # peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    # peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    # ppo model
    ppo_model_outputs = ppo_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    ppo_model_text_output = tokenizer.decode(ppo_model_outputs[0], skip_special_tokens=True)

    original_model_summaries.append(original_model_text_output)
    # instruct_model_summaries.append(instruct_model_text_output)
    # peft_model_summaries.append(peft_model_text_output)
    ppo_model_summaries.append(ppo_model_text_output)    

# ## Evalute the Full Dataset
# 
# The file called "diag-summary-training-results-with-peft.csv" contains a pre-populated list of all model results which we can use to evaluate on a larger section of data. The results show substantial improvement in all ROUGE metrics!

# In[50]:


import evaluate

rouge = evaluate.load('rouge')

# In[51]:


import pandas as pd
results = pd.read_csv("diag-summary-training-results-with-peft.csv")
human_baseline_summaries = results['human_baseline_summaries'].values
original_model_summaries = results['original_model_summaries'].values
instruct_model_summaries = results['instruct_model_summaries'].values
peft_model_summaries     = results['peft_model_summaries'].values

# In[52]:


original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
original_model_results

# In[53]:


instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
instruct_model_results

# In[54]:


peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
peft_model_results

# In[55]:


ppo_model_results = rouge.compute(
    predictions=ppo_model_summaries,
    references=human_baseline_summaries[0:len(ppo_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
ppo_model_results

# # Calculate improvement of PPO over original

# In[33]:


import numpy as np
improvement = (np.array(list(ppo_model_results.values())) - np.array(list(original_model_results.values())))
for key, value in zip(ppo_model_results.keys(), improvement):
    print(f'{key} absolute percentage improvement of peft model over original model: {value*100:.2f}%')

# # Calculate improvement of PPO over Instruct

# In[34]:


import numpy as np
improvement = (np.array(list(ppo_model_results.values())) - np.array(list(instruct_model_results.values())))
for key, value in zip(ppo_model_results.keys(), improvement):
    print(f'{key} absolute percentage improvement of peft model over instruct model: {value*100:.2f}%')

# # Calculate improvement of PPO over PEFT

# In[35]:


improvement = (np.array(list(ppo_model_results.values())) - np.array(list(peft_model_results.values())))
for key, value in zip(ppo_model_results.keys(), improvement):
    print(f'{key} absolute percentage improvement of peft model over instruct model: {value*100:.2f}%')

# # TODO:  Add something like this to show toxicity/reward before and after PPO
# 
# ![](https://camo.githubusercontent.com/10d611b486cb332ebd30853740f4b74d9d2573b2ec6c033aaefed173fc463065/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f74726c2d696e7465726e616c2d74657374696e672f6578616d706c652d696d616765732f7265736f6c76652f6d61696e2f696d616765732f7461626c655f696d64625f707265766965772e706e67)

# In[ ]:



