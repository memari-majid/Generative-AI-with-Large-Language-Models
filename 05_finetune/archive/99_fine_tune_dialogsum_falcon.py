#!/usr/bin/env python
# coding: utf-8

# # Fine-tune the Instructor-Model for Dialogue Summarization

# <a name='1'></a>
# ## Set up Kernel and Required Dependencies

# First, check that the correct kernel is chosen.
# 
# <img src="img/kernel_set_up.png" width="300"/>
# 
# You can click on that to see and check the details of the image, kernel, and instance type.
# 
# <img src="img/w3_kernel_and_instance_type.png" width="600"/>

# In[2]:


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

# In[3]:


%store -r model_checkpoint

# In[4]:


try:
    model_checkpoint
except NameError:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("[ERROR] Please run the notebooks in the PREPARE section before you continue.")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# In[5]:


print(model_checkpoint)

# In[6]:


%store -r local_data_processed_path

# In[7]:


try:
    local_data_processed_path
except NameError:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("[ERROR] Please run the notebooks in the PREPARE section before you continue.")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# In[8]:


print(local_data_processed_path)

# # Load Packages

# In[9]:


import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, GenerationConfig
from datasets import load_dataset
import datasets
import torch
import time
import evaluate
import numpy as np
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load HuggingFace Model
# 
# We can load the pre-trained Flan-T5 model directly from HuggingFace. Notice that we will be using the [base version](https://huggingface.co/google/flan-t5-base) of flan. This model version has ~247 million model parameters which makes it small compared to other LLMs. For higher quality results, we recommend looking into the larger versions of this model.

# In[10]:


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

model = AutoModelForCausalLM.from_pretrained(model_checkpoint, 
                                             trust_remote_code=True, 
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto", # place shards automatically
                                             # load_in_8bit=True, # fails
                                             # quantization_config=bnb_config
                                            )

# In[11]:


params = sum(p.numel() for p in model.parameters())
print(f'Total Number of Model Parameters: {params}')

# # Load the Processed Data
# The DialogSum dataset which we processed already can loaded directly from our local directory. There are ~15k examples of dialogue in this dataset with associated human summarizations of these datasets

# In[12]:


tokenized_dataset = load_dataset(
    local_data_processed_path,
    data_files={'train': 'train/*.parquet', 'test': 'test/*.parquet', 'validation': 'validation/*.parquet'}
).with_format("torch")

tokenized_dataset

# # Test the Model with Zero-Shot Prompts BEFORE Fine-Tuning
# 
# In the example below, we highlight how the summarization capability of the model is lacking compared to the baseline summary provided in the dataset. You can see that the model struggles to summarize the dialogue compared to the baseline summary, but it does pull out some important information from the text which indicates the model can be fine tuned to the task at hand.

# In[13]:


from random import randint

# Load dataset from the hub
test_dataset = load_dataset("knkarthick/dialogsum", split="test")

# select a random test sample
sample = test_dataset[randint(0, len(test_dataset))]

# format sample
#prompt_template = f"Summarize the following conversation.\n\n{{dialogue}}\n\nSummary: "
prompt_template = f"Summarize the chat dialogue:\n{{dialogue}}\n---\nSummary:\n"

sample_prompt = prompt_template.format(dialogue=sample["dialogue"])
human_baseline_completion = sample["summary"]

print('Prompt:')
print('--------------------------')
print(sample_prompt)
print('--------------------------')
print(f'### Human Baseline Completion\n: {human_baseline_completion}')

# In[14]:


def remove_prompt_and_decode(input_ids, completion_ids):
    # remove the input_ids from the completion_ids, then decode just the result (without the prompt)
    return tokenizer.decode(completion_ids[:, input_ids.shape[1]:][0], skip_special_tokens=True)

# In[15]:


input_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids

completion_ids = model.generate(input_ids=input_ids.to(DEVICE), 
                            generation_config=GenerationConfig(
                                max_new_tokens=500, 
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id
                            ))

model_completion = remove_prompt_and_decode(input_ids, completion_ids)

print('Prompt:\n')
print('--------------------------')
print(sample_prompt)
print('--------------------------')
print(f'### Human Baseline Completion:\n{human_baseline_completion}')
print()
print(f'### Model Completion:\n{model_completion}')

# In[16]:


# pipeline = transformers.pipeline(
#     "text-generation", # summarization? # https://huggingface.co/tiiuae/falcon-40b/discussions/8
#     model=model,
#     tokenizer=tokenizer,
# )

# In[17]:


# pipeline(sample_prompt,
#          max_new_tokens=500)

# # Fine-tune the model
# 
# Now that the dataset is preprocessed, we can utilize the built-in HuggingFace `Trainer` class to fine tune our model to the task at hand. Please note that training this full model takes a few hours on a GPU, so for the sake of time, a checkpoint for a model which has been trained on 10 epochs without downsampling has been provided. If you have time to experiment on fully training the model yourself, please see the inline comments for how to change up the code. If you are looking to train on a GPU machine, we have used a `ml.g5.xlarge` instance for the checkpoint provided as a place to start.

# In[18]:


# for the sake of time in the lab, we will subsample our dataset
# if you want to take the time to train a model fully, feel free to alter this subsampling to create a larger dataset
sample_tokenized_dataset = tokenized_dataset.filter(lambda example, indice: indice % 100 == 0, with_indices=True)
print(sample_tokenized_dataset.shape)
sample_tokenized_dataset['train'][0]['input_ids']

# In[19]:


output_dir = f'./diag-summary-training-{str(int(time.time()))}'
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5, 
    num_train_epochs=1,
    # num_train_epochs=10, # Use a higher number of epochs when you are not in the lab and have more time to experiment
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sample_tokenized_dataset['train'],
    eval_dataset=sample_tokenized_dataset['validation'],
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),    
)

# # This will take a few minutes even after it appears to finish

# In[20]:


trainer.train()

# In[21]:


trainer.evaluate()

# In[ ]:


supervised_fine_tuned_model_path = "./falcon-dialogue-summary-checkpoint"

trainer.save_model(supervised_fine_tuned_model_path)

# # Load the Original and Fine-Tuned Models to Compare
# 
# Once the model has finished training, we will load both the original model from HuggingFace and the fune tuned model to do some qualitative and quantitative comparisions.

# In[ ]:


model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
    device_map="auto", # place shards automatically
    # load_in_8bit=True, # fails
    # quantization_config=bnb_config
)

tuned_model = AutoModelForCausalLM.from_pretrained(
    supervised_fine_tuned_model_path,
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
    device_map="auto", # place shards automatically
    # load_in_8bit=True, # fails
    # quantization_config=bnb_config
)

# In[ ]:


# !aws s3 cp --recursive s3://dsoaws/models/flan-dialogue-summary-checkpoint/ ./flan-dialogue-summary-checkpoint/

# In[ ]:


# # supervised_fine_tuned_model_path = f"./{output_dir}/<put-your-checkpoint-dir-here>"

# tuned_model = AutoModelForSeq2SeqLM.from_pretrained(supervised_fine_tuned_model_path)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# In[ ]:


# %store supervised_fine_tuned_model_path

# # Qualitative Results with Zero Shot Inference AFTER Fine-Tuning
# 
# As with many GenAI applications, a qualitative approach where you ask yourself the question "is my model behaving the way it is supposed to?" is usually a good starting point. In the example below (the same one we started this notebook with), you can see how the fine-tuned model is able to create a reasonable summary of the dialogue compared to the original inability to understand what is being asked of the model.

# In[ ]:


from datasets import concatenate_datasets
dataset = load_dataset("knkarthick/dialogsum")
#[dataset['train'], dataset['test'], dataset['validation']])
print(dataset)

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# dataset = load_dataset('./data-summarization/')
# dataset

# In[ ]:


prompt_template = f"Summarize the chat dialogue:\n{{dialogue}}\n---\nSummary:\n"

# template dataset to add prompt to each sample
def tokenize_prompt(sample):
    prompt = prompt_template.format(dialogue=sample["dialogue"],
                                                   summary=sample["summary"],
                                                   eos_token=tokenizer.eos_token)
    return tokenizer(prompt)

# In[ ]:


tokenized_prompt_datasets = dataset.map(tokenize_prompt) #, batched=True)
tokenized_prompt_datasets

# In[ ]:


from random import randint

idx = randint(0, len(dataset))

prompt = tokenized_prompt_datasets['test'][idx]['dialogue']
print(prompt)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(input_ids)

completion_ids = model.generate(
    input_ids=input_ids.to(DEVICE),
    generation_config=GenerationConfig(
        max_new_tokens=500, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id)
)
print(completion_ids)
print(tokenizer.decode(completion_ids[0], skip_special_tokens=True))

finetuned_completion_ids = tuned_model.generate(
    input_ids=input_ids.to(DEVICE),
    generation_config=GenerationConfig(
        max_new_tokens=500, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id)
)

# In[ ]:


#decoded_original_outputs = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
#decoded_tuned_outputs = tokenizer.decode(tuned_outputs[0], skip_special_tokens=True)

completion = remove_prompt_and_decode(input_ids, completion_ids)
finetuned_completion = remove_prompt_and_decode(input_ids, finetuned_completion_ids)

print('--------------------------')
print('--------------------------')
print(f'Prompt:')
print('--------------------------')
print('--------------------------')
print(prompt)
print('--------------------------')
print(f'### Model completion:\n{completion}')
print('--------------------------')
print(f'### Fine-tuned completion:\n{finetuned_completion}')
print('--------------------------')
print(f'### Human Baseline Completion:\n{human_baseline_completion}')
print('--------------------------')

# # Quatitative Results with ROGUE Metric
# 
# The [ROUGE metric](https://en.wikipedia.org/wiki/ROUGE_(metric)) helps quantify the validity of summarizations produced by models. It compares summarizations to a "baseline" summary which is usually created by a human. While not perfect, it does give an indication to the overall increase in summarization effectiveness that we have accomplished by fine-tuning.

# In[ ]:


rouge = evaluate.load('rouge')

# ## Evaluate a Subsection of Summaries

# In[ ]:


# again, for the sake of time, we will only be generating a few summaries with each model
# outside of the lab, a good exercise is to increase the number of validation summaries generated
input_ids = tokenized_dataset['test'][0:10]['input_ids']
#baseline_summaries = tokenized_dataset['test'][0:10]['labels']

# decode the original summaries
# human_baseline_summaries = []
# for base_summary in baseline_summaries:
#     human_baseline_summaries.append(tokenizer.decode(base_summary, skip_special_tokens=True))

# generate the summaries
original_outputs = model.generate(input_ids.to(DEVICE), 
                                  GenerationConfig(
                                      max_new_tokens=200,
                                      pad_token_id=tokenizer.eos_token_id
                                  ))
tuned_outputs = tuned_model.generate(input_ids.to(DEVICE), 
                                     GenerationConfig(
                                         max_new_tokens=200,
                                         pad_token_id=tokenizer.eos_token_id
                                    ))

# In[ ]:


# store the summaries in lists
original_model_summaries = []
tuned_model_summaries = []

# decode all the summaries
for original_summary, tuned_summary in zip(original_outputs, tuned_outputs):
    original_model_summaries.append(tokenizer.decode(original_summary, skip_special_tokens=True))
    tuned_model_summaries.append(tokenizer.decode(tuned_summary, skip_special_tokens=True))

# In[ ]:


original_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries,
    use_aggregator=True,
    use_stemmer=True,
)

# In[ ]:


tuned_results = rouge.compute(
    predictions=tuned_model_summaries,
    references=human_baseline_summaries,
    use_aggregator=True,
    use_stemmer=True,
)

# In[ ]:


original_results

# In[ ]:


tuned_results

# ## Evalute the Full Dataset
# 
# The file called "diag-summary-training-results.csv" contains a pre-populated list of all model results which we can use to evaluate on a larger section of data. The results show substantial improvement in all ROUGE metrics!

# In[ ]:


import pandas as pd
results = pd.read_csv("diag-summary-training-results.csv")
original_model_summaries = results['original_model_summaries'].values
tuned_model_summaries = results['tuned_model_summaries'].values
human_baseline_summaries = results['human_baseline_summaries'].values

# In[ ]:


original_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

# In[ ]:


tuned_results = rouge.compute(
    predictions=tuned_model_summaries,
    references=human_baseline_summaries[0:len(tuned_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

# In[ ]:


original_results

# In[ ]:


tuned_results

# In[ ]:


improvement = (np.array(list(tuned_results.values())) - np.array(list(original_results.values())))
for key, value in zip(tuned_results.keys(), improvement):
    print(f'{key} absolute percentage improvement after instruct fine-tuning: {value*100:.2f}%')

# # Release Resources

# In[ ]:


# %%html

# <p><b>Shutting down your kernel for this notebook to release resources.</b></p>
# <button class="sm-command-button" data-commandlinker-command="kernelmenu:shutdown" style="display:none;">Shutdown Kernel</button>
        
# <script>
# try {
#     els = document.getElementsByClassName("sm-command-button");
#     els[0].click();
# }
# catch(err) {
#     // NoOp
# }    
# </script>

# In[ ]:



