#!/usr/bin/env python
# coding: utf-8

# # Feature Transformation in this Notebook
# 
# In this notebook, we convert raw text into tokenized inputs which are ready to be ingested by a HuggingFace training script. It is important to separate this tokenizing and prompt creation step of the process from training because it will allow you to tailor the compute which is most efficient for each step of the process. For instance a low-cost CPU processor is generally the best bet for the preparation section of the workflow while a higher cost GPU instance is best for model training.

# # Tested on ml.m5.2xlarge

# <a name='1'></a>
# ## Set up Kernel and Required Dependencies

# In[ ]:


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

# In[7]:


from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import os
import time

# ## Ensure the Base Dataset is Downloaded

# In[ ]:


from datasets import concatenate_datasets
dataset = load_dataset("knkarthick/dialogsum")
dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])
!mkdir data-summarization
dataset = dataset.train_test_split(0.5, seed=1234)
dataset['test'].to_csv('./data-summarization/dialogsum-1.csv', index=False)
dataset['train'].to_csv('./data-summarization/dialogsum-2.csv', index=False)

# ## Load the Tokenizer and HuggingFace Dataset

# In[8]:


model_checkpoint='google/flan-t5-base'

# In[9]:


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
dataset = load_dataset('./data-summarization/')
dataset

# ## Explore an Example Prompt

# In[10]:


idx = 0
diag = dataset['train'][idx]['dialogue']
baseline_human_summary = dataset['train'][idx]['summary']

prompt = f'Summarize the following conversation.\n\n{diag}\n\nSummary:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

print(f'Prompt:\n--------------------------\n{prompt}\n--------------------------')
print(f'Baseline human summary : {baseline_human_summary}')

# ## Tokenize the Dataset

# In[11]:


def tokenize_function(example):
    prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    inp = [prompt + i + end_prompt for i in example["dialogue"]]
    example['input_ids'] = tokenizer(inp, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])

# ## Wrap the preprocessing into a repeatable function

# In[12]:


def tokenize_function(example):
    prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    inp = [prompt + i + end_prompt for i in example["dialogue"]]
    example['input_ids'] = tokenizer(inp, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example

def transform_dataset(input_data,
                      output_data,
                      huggingface_model_name,
                      train_split_percentage,
                      test_split_percentage,
                      validation_split_percentage,
                      ):

    # load in the original dataset
    dataset = load_dataset(input_data)
    print(f'Dataset loaded from path: {input_data}\n{dataset}')
    
    # Load the tokenizer
    print(f'Loading the tokenizer for the model {huggingface_model_name}')
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
    
    # make train test validation split
    train_testvalid = dataset['train'].train_test_split(1 - train_split_percentage, seed=1234)
    test_valid = train_testvalid['test'].train_test_split(test_split_percentage / (validation_split_percentage + test_split_percentage), seed=1234)
    train_test_valid_dataset = DatasetDict(
        {
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'validation': test_valid['train']
        }
    )
    print(f'Dataset after splitting:\n{train_test_valid_dataset}')
    
    # tokenize the dataset
    print(f'Tokenizing the dataset...')
    tokenized_datasets = train_test_valid_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])
    print(f'Tokenizing complete!')
    
    # create directory for drop
    os.makedirs(f'{output_data}/train/', exist_ok=True)
    os.makedirs(f'{output_data}/test/', exist_ok=True)
    os.makedirs(f'{output_data}/validation/', exist_ok=True)
    file_root = str(int(time.time()*1000))
    
    # save the dataset to disk
    print(f'Writing the dataset to {output_data}')
    tokenized_datasets['train'].to_parquet(f'./{output_data}/train/{file_root}.parquet')
    tokenized_datasets['test'].to_parquet(f'./{output_data}/test/{file_root}.parquet')
    tokenized_datasets['validation'].to_parquet(f'./{output_data}/validation/{file_root}.parquet')
    print('Preprocessing complete!')

# In[13]:


def process(args):

    print(f"Listing contents of {args.input_data}")
    dirs_input = os.listdir(args.input_data)
    for file in dirs_input:
        print(file)

    transform_dataset(input_data=args.input_data, #'./data-summarization/',
                      output_data=args.output_data, #'./data-summarization-processed/',
                      huggingface_model_name=args.model_checkpoint, #model_checkpoint,
                      train_split_percentage=args.train_split_percentage, #0.90
                      test_split_percentage=args.test_split_percentage, #0.05
                      validation_split_percentage=args.validation_split_percentage, #0.05
                     )

    print(f"Listing contents of {args.output_data}")
    dirs_output = os.listdir(args.output_data)
    for file in dirs_output:
        print(file)

# # Process the Dataset Locally

# In[14]:


class Args:
    input_data: str
    output_data: str
    model_checkpoint: str
    train_split_percentage: float
    test_split_percentage: float
    validation_split_percentage: float

args = Args()

args.model_checkpoint = model_checkpoint
args.input_data = './data-summarization'
args.output_data = './data-summarization-processed'
args.train_split_percentage = 0.9
args.test_split_percentage = 0.05
args.validation_split_percentage = 0.05

# remove any data that is already saved locally
if os.path.isdir(args.output_data):
    import shutil
    shutil.rmtree(args.output_data)

process(args)

# ## Ensure the dataset can be loaded correctly

# In[15]:


dataset = load_dataset(
    './data-summarization-processed/',
    data_files={'train': 'train/*.parquet', 'test': 'test/*.parquet', 'validation': 'validation/*.parquet'}
)
dataset

# In[ ]:



