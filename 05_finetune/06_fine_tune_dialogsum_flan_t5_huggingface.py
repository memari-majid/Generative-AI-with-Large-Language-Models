#!/usr/bin/env python
# coding: utf-8

# # Fine-tune the Instructor-Model for Dialogue Summarization

# # Tested on ml.m5.2xlarge

# <a name='1'></a>
# ## Set up Kernel and Required Dependencies

# In[2]:


# %pip install --disable-pip-version-check \
#     torch==2.0.1 \
#     transformers==4.34.1 \
#     datasets==2.12.0 \
#     accelerate==0.23.0 \
#     evaluate==0.4.0 \
#     py7zr==0.20.4 \
#     sentencepiece==0.1.99 \
#     rouge_score==0.1.2 \
#     loralib==0.1.1 \
#     peft==0.4.0 \
#     trl==0.7.2

# In[3]:


model_checkpoint='google/flan-t5-base'

# In[4]:


# this directory is created in the previous notebook
local_data_processed_path = './data-summarization-processed/'

# # Load Packages

# In[5]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, GenerationConfig
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

# In[6]:


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# In[7]:


params = sum(p.numel() for p in model.parameters())
print(f'Total Number of Model Parameters: {params}')

# # Load the Processed Data

# # Load Dataset
# 
# The DialogSum dataset which we processed already can loaded directly from our local directory. There are ~15k examples of dialogue in this dataset with associated human summarizations of these datasets

# In[8]:


tokenized_dataset = load_dataset(
    local_data_processed_path,
    data_files={'train': 'train/*.parquet', 'test': 'test/*.parquet', 'validation': 'validation/*.parquet'}
).with_format("torch")
tokenized_dataset

# # Test the Model with Zero-Shot Prompts BEFORE Fine-Tuning
# 
# In the example below, we highlight how the summarization capability of the model is lacking compared to the baseline summary provided in the dataset. You can see that the model struggles to summarize the dialogue compared to the baseline summary, but it does pull out some important information from the text which indicates the model can be fine tuned to the task at hand.

# In[9]:


idx = 2
diag = tokenizer.decode(tokenized_dataset['test'][idx]['input_ids'], skip_special_tokens=True)
model_input = tokenizer(diag, return_tensors="pt").input_ids
summary = tokenizer.decode(tokenized_dataset['test'][idx]['labels'], skip_special_tokens=True)

original_outputs = model.to('cpu').generate(model_input, GenerationConfig(max_new_tokens=200))
original_text_output = tokenizer.decode(original_outputs[0], skip_special_tokens=True)

diag_print = diag.replace(' #',' \n#')
print(f"Prompt:\n--------------------------\n{diag_print}\n--------------------------")
print(f'\nOriginal Model Response: {original_text_output}')
print(f'Baseline Summary : {summary}')

# # Fine-tune the instructor model
# 
# Now that the dataset is preprocessed, we can utilize the built-in HuggingFace `Trainer` class to fine tune our model to the task at hand. Please note that training this full model takes a few hours on a GPU, so for the sake of time, a checkpoint for a model which has been trained on 10 epochs without downsampling has been provided. If you have time to experiment on fully training the model yourself, please see the inline comments for how to change up the code. If you are looking to train on a GPU machine, we have used a `ml.g5.xlarge` instance for the checkpoint provided as a place to start.

# In[10]:


# for the sake of time in the lab, we will subsample our dataset
# if you want to take the time to train a model fully, feel free to alter this subsampling to create a larger dataset
sample_tokenized_dataset = tokenized_dataset.filter(lambda example, indice: indice % 100 == 0, with_indices=True)

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
    eval_dataset=sample_tokenized_dataset['validation']
)

# In[11]:


trainer.train()

# # Load the Trained Model and Original Model
# 
# Once the model has finished training, we will load both the original model from HuggingFace and the fine-tuned model to do some qualitative and quantitative comparisions.

# In[12]:


!aws s3 cp --recursive s3://dsoaws/models/flan-dialogue-summary-checkpoint/ ./flan-dialogue-summary-checkpoint/

# In[13]:


# if you have trained your own model and want to check it out compared to ours, change the line of code
# below to contain your checkpoint directory

supervised_fine_tuned_model_path = "./flan-dialogue-summary-checkpoint"
# supervised_fine_tuned_model_path = f"./{output_dir}/<put-your-checkpoint-dir-here>"

tuned_model = AutoModelForSeq2SeqLM.from_pretrained(supervised_fine_tuned_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# In[14]:


%store supervised_fine_tuned_model_path

# # Qualitative Results with Zero Shot Inference AFTER Fine-Tuning
# 
# As with many GenAI applications, a qualitative approach where you ask yourself the question "is my model behaving the way it is supposed to?" is usually a good starting point. In the example below (the same one we started this notebook with), you can see how the fine-tuned model is able to create a reasonable summary of the dialogue compared to the original inability to understand what is being asked of the model.

# In[15]:


idx = 2
diag = tokenizer.decode(tokenized_dataset['test'][idx]['input_ids'], skip_special_tokens=True)
model_input = tokenizer(diag, return_tensors="pt").input_ids
summary = tokenizer.decode(tokenized_dataset['test'][idx]['labels'], skip_special_tokens=True)

original_outputs = model.to('cpu').generate(
    model_input,
    GenerationConfig(max_new_tokens=200, num_beams=1),
)
outputs = tuned_model.to('cpu').generate(
    model_input,
    GenerationConfig(max_new_tokens=200, num_beams=1,),
)
text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

diag_print = diag.replace(' #',' \n#')
print(f"Prompt:\n--------------------------\n{diag_print}\n--------------------------")
print(f'Flan-T5 response: {original_text_output}')
print(f'Our instruct-tuned response (on top of Flan-T5): {text_output}')
print(f'Baseline summary from original dataset: {summary}')

# # Quantitative Results with ROUGE Metric
# 
# The [ROUGE metric](https://en.wikipedia.org/wiki/ROUGE_(metric)) helps quantify the validity of summarizations produced by models. It compares summarizations to a "baseline" summary which is usually created by a human. While not perfect, it does give an indication to the overall increase in summarization effectiveness that we have accomplished by fine-tuning.

# In[16]:


rouge = evaluate.load('rouge')

# ## Evaluate a Subsection of Summaries

# In[17]:


# again, for the sake of time, we will only be generating a few summaries with each model
# outside of the lab, a good exercise is to increase the number of validation summaries generated
dialogues = tokenized_dataset['test'][0:10]['input_ids']
baseline_summaries = tokenized_dataset['test'][0:10]['labels']

# decode the original summaries
human_baseline_summaries = []
for base_summary in baseline_summaries:
    human_baseline_summaries.append(tokenizer.decode(base_summary, skip_special_tokens=True))

# generate the summaries
original_outputs = model.generate(dialogues, GenerationConfig(max_new_tokens=200))
tuned_outputs = tuned_model.generate(dialogues, GenerationConfig(max_new_tokens=200))

# In[18]:


# store the summaries in lists
original_model_summaries = []
tuned_model_summaries = []

# decode all the summaries
for original_summary, tuned_summary in zip(original_outputs, tuned_outputs):
    original_model_summaries.append(tokenizer.decode(original_summary, skip_special_tokens=True))
    tuned_model_summaries.append(tokenizer.decode(tuned_summary, skip_special_tokens=True))

# In[19]:


original_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries,
    use_aggregator=True,
    use_stemmer=True,
)

# In[20]:


tuned_results = rouge.compute(
    predictions=tuned_model_summaries,
    references=human_baseline_summaries,
    use_aggregator=True,
    use_stemmer=True,
)

# In[21]:


original_results

# In[22]:


tuned_results

# ## Evalute the Full Dataset
# 
# The file called "diag-summary-training-results.csv" contains a pre-populated list of all model results which we can use to evaluate on a larger section of data. The results show substantial improvement in all ROUGE metrics!

# In[23]:


import pandas as pd
results = pd.read_csv("diag-summary-training-results.csv")
original_model_summaries = results['original_model_summaries'].values
tuned_model_summaries = results['tuned_model_summaries'].values
human_baseline_summaries = results['human_baseline_summaries'].values

# In[24]:


original_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

# In[25]:


tuned_results = rouge.compute(
    predictions=tuned_model_summaries,
    references=human_baseline_summaries[0:len(tuned_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

# In[26]:


original_results

# In[27]:


tuned_results

# In[28]:


improvement = (np.array(list(tuned_results.values())) - np.array(list(original_results.values())))
for key, value in zip(tuned_results.keys(), improvement):
    print(f'{key} absolute percentage improvement after instruct fine-tuning: {value*100:.2f}%')
