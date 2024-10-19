#!/usr/bin/env python
# coding: utf-8

# # Fine-Tune a Generative AI Model for Dialogue Summarization

# In this notebook we will see how to fine tune an existing LLM from HuggingFace for enhanced dialogue summarization. We will be using the [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) model as it provides a high quality instruction tuned model at various sizes. Flan-T5 can summarize text out of the box, but in this notebook we will see how fine-tuning on a high quality dataset can improve its performance for a specific task. Specifically, we will be using the [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) dataset from HuggingFace which contains chunks of dialogue and associated summarizations of the dialogue.

# ## Setup
# 
# First up, lets make sure we install some libraries which are needed for this notebook. After the installation, we will import the necessary packages for the notebook

# In[ ]:


%pip install torch==1.13.1
%pip install torchdata
%pip install transformers==4.27.2 --quiet
%pip install torch==1.13.1 --quiet
%pip install py7zr==0.20.4 --quiet
%pip install datasets==2.9.0 --quiet
%pip install sentencepiece==0.1.97 --quiet
%pip install evaluate==0.4.0 --quiet
%pip install accelerate==0.17.0
%pip install rouge_score==0.1.2 --quiet
%pip install loralib==0.1.1 --quiet

# In[1]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, GenerationConfig
from datasets import load_dataset
import datasets
import torch
import time
import evaluate
import numpy as np
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load Flan-T5 Model
# 
# We can load the pre-trained Flan-T5 model directly from HuggingFace. Notice that we will be using the [small version](https://huggingface.co/google/flan-t5-small) of flan.

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
original_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", device_map="auto")

# In[ ]:


params = sum(p.numel() for p in original_model.parameters())
print(f'Total number of model parameters: {params}')

# # Load Dataset
# 
# The DialogSum dataset can also be loaded directly from HuggingFace. There are ~15k examples of dialogue in this dataset with associated human summarizations of these datasets

# In[14]:


dataset = load_dataset("knkarthick/dialogsum")
dataset

# # Test the Model with Zero-Shot Prompts Before Tuning
# 
# In the example below, we highlight how the summarization capability of the model is lacking compared to the baseline summary provided in the dataset. You can see that the model struggles to summarize the dialogue compared to the baseline summary, but it does pull out some important information from the text which indicates the model can be fine tuned to the task at hand.

# In[15]:


idx = 20
diag = dataset['test'][idx]['dialogue']
baseline_human_summary = dataset['test'][idx]['summary']

prompt = f'Summarize the following conversation.\n\n{diag}\n\nSummary:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

original_outputs = original_model.generate(input_ids, GenerationConfig(max_new_tokens=200))
original_text_output = tokenizer.decode(original_outputs[0], skip_special_tokens=True)

print(f'Prompt:\n--------------------------\n{prompt}\n--------------------------')
print(f'Baseline human summary : {baseline_human_summary}')
print(f'\nOriginal Flan-T5 summary: {original_text_output}')

# # Preprocessing
# 
# To preprocess the dataset, we need to append a useful prompt to the start and end of each dialogue set then tokenize the words with HuggingFace. The output dataset will be ready for fine tuning in the next step.

# In[16]:


def tokenize_function(example):
    prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    inp = [prompt + i + end_prompt for i in example["dialogue"]]
    example['input_ids'] = tokenizer(inp, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])

# In[17]:


tokenized_datasets

# # Fine Tuning with HuggingFace Trainer
# 
# Now that the dataset is preprocessed, we can utilize the built-in HuggingFace `Trainer` class to fine tune our model to the task at hand. Please note that training this full model takes a few hours on a GPU, so for the sake of time, a checkpoint for a model which has been trained on 10 epochs without downsampling has been provided. If you have time to experiment on fully training the model yourself, please see the inline comments for how to change up the code. If you are looking to train on a GPU machine, we have used a `ml.g5.xlarge` instance for the checkpoint provided as a place to start.

# In[18]:


# for the sake of time in the lab, we will subsample our dataset
# if you want to take the time to train a model fully, feel free to alter this subsampling to create a larger dataset
# the line below can be completely removed to remove the subsampling
#tokenized_datasets = tokenized_datasets.filter(lambda example, indice: indice % 100 == 0, with_indices=True)

# In[19]:


output_dir = f'./diag-summary-training-{str(int(time.time()))}'
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

# In[20]:


# trainer.train()

# # Load the Trained Model and Original Model
# 
# Once the model has finished training, we will load both the original model from HuggingFace and the fune tuned model to do some qualitative and quantitative comparisions.

# In[21]:


!aws s3 cp --recursive s3://dsoaws/models/flan-dialogue-summary-checkpoint/ ./flan-dialogue-summary-checkpoint/

# # Show size of instruct model

# In[22]:


!ls -al ./flan-dialogue-summary-checkpoint/pytorch_model.bin

# In[23]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, GenerationConfig

# if you have trained your own model and want to check it out compared to ours, 
# uncomment the line of code below to contain your checkpoint directory
#instruct_model = AutoModelForSeq2SeqLM.from_pretrained(f"./{output_dir}/<put-your-checkpoint-dir-here>")

instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./flan-dialogue-summary-checkpoint", device_map="auto")

original_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", device_map="auto")

# # Qualitative Results
# 
# As with many GenAI applications, a qualitative approach where you ask yourself the question "is my model behaving the way it is supposed to?" is usually a good starting point. In the example below (the same one we started this notebook with), you can see how the fine-tuned model is able to create a reasonable summary of the dialogue compared to the original inability to understand what is being asked of the model.

# In[24]:


idx = 20
diag = dataset['test'][idx]['dialogue']
baseline_human_summary = dataset['test'][idx]['summary']

prompt = f'Summarize the following conversation.\n\n{diag}\n\nSummary:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

print(f'Prompt:\n--------------------------\n{prompt}\n--------------------------')
print(f'Baseline human summary from original dataset: {baseline_human_summary}')
print(f'Original Flan-T5 summary: {original_model_text_output}')
print(f'Instruct model summary: {instruct_model_text_output}')

# # Quatitative Results with ROGUE Metric
# 
# The [ROUGE metric](https://en.wikipedia.org/wiki/ROUGE_(metric)) helps quantify the validity of summarizations produced by models. It compares summarizations to a "baseline" summary which is usually created by a human. While not perfect, it does give an indication to the overall increase in summarization effectiveness that we have accomplished by fine-tuning.

# In[25]:


rouge = evaluate.load('rouge')

# ## Evaluate a Subsection of Summaries

# In[26]:


# again, for the sake of time, we will only be generating 10 summarizations with each model
# outside of the lab, a good exercise is to increase the number of validation summaries generated
dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']

original_model_summaries = []
instruct_model_summaries = []

for ind, diag in enumerate(dialogues):
    prompt = f'Summarize the following conversation.\n\nConversation:\n{diag}\n\nSummary:'
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
    original_model_summaries.append(original_model_text_output)

    instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
    instruct_model_summaries.append(instruct_model_text_output)

# In[27]:


original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

# In[28]:


instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

# In[29]:


original_model_results

# In[30]:


instruct_model_results

# ## Evalute the Full Dataset
# 
# The file called "diag-summary-training-results.csv" contains a pre-populated list of all model results which we can use to evaluate on a larger section of data. The results show substantial improvement in all ROUGE metrics!

# In[31]:


import pandas as pd
results = pd.read_csv("diag-summary-training-results-with-peft.csv")

human_baseline_summaries = results['human_baseline_summaries'].values
original_model_summaries = results['original_model_summaries'].values
instruct_model_summaries = results['instruct_model_summaries'].values
#peft_model_summaries = results['peft_model_summaries'].values

# In[32]:


original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

# In[33]:


instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

# In[34]:


original_model_results

# In[35]:


instruct_model_results

# In[36]:


improvement = (np.array(list(instruct_model_results.values())) - np.array(list(original_model_results.values())))
for key, value in zip(instruct_model_results.keys(), improvement):
    print(f'{key} absolute percentage improvement of instruct model over human baseline: {value*100:.2f}%')

# # PEFT

# In[37]:


%pip install git+https://github.com/huggingface/peft.git

# In[38]:


# re-importing as the rest of this notebook will likely move to a new notebook.  you're welcome!

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, GenerationConfig
from datasets import load_dataset
import datasets
import torch
import time
import evaluate
import numpy as np
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load base Flan-T5 model and tokenizer
# 
# We can load the pre-trained Flan-T5 model directly from HuggingFace. Notice that we will be using the [base version](https://huggingface.co/google/flan-t5-base) of flan to create the PEFT adapter.  We will compare this to an fully-fine tuned instruct model.

# In[39]:


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

original_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# # Add PEFT layer/parameter adapters
# Note the rank (`r`) hyper-parameter below.

# In[40]:


from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# Define LoRA Config
lora_config = LoraConfig(
 r=32, # rank
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)

# Add LoRA adapter layers/parameters 
peft_model = get_peft_model(original_model, lora_config)
peft_model.print_trainable_parameters()

# In[41]:


from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=peft_model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# # Load Dataset
# 
# The DialogSum dataset can also be loaded directly from HuggingFace. There are ~15k examples of dialogue in this dataset with associated human summarizations of these datasets

# In[42]:


dataset = load_dataset("knkarthick/dialogsum")
dataset

# # Test the Model with Zero-Shot Prompts Before Tuning
# 
# In the example below, we highlight how the summarization capability of the model is lacking compared to the baseline summary provided in the dataset. You can see that the model struggles to summarize the dialogue compared to the baseline summary, but it does pull out some important information from the text which indicates the model can be fine tuned to the task at hand.

# In[43]:


idx = 20
diag = dataset['test'][idx]['dialogue']
baseline_human_summary = dataset['test'][idx]['summary']

prompt = f'Summarize the following conversation.\n\n{diag}\n\nSummary:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

original_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
original_text_output = tokenizer.decode(original_outputs[0], skip_special_tokens=True)

print(f'Prompt:\n--------------------------\n{prompt}\n--------------------------')
print(f'Baseline human summary : {baseline_human_summary}')
print(f'Original Flan-T5 summary: {original_text_output}')

# # Preprocessing
# 
# To preprocess the dataset, we need to append a useful prompt to the start and end of each dialogue set then tokenize the words with HuggingFace. The output dataset will be ready for fine tuning in the next step.

# In[44]:


def tokenize_function(example):
    prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    inp = [prompt + i + end_prompt for i in example["dialogue"]]
    example['input_ids'] = tokenizer(inp, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])

# In[45]:


# for the sake of time in the lab, we will subsample our dataset
# if you want to take the time to train a model fully, feel free to alter this subsampling to create a larger dataset
# the line below can be completely removed to remove the subsampling
#tokenized_datasets = tokenized_datasets.filter(lambda example, indice: indice % 100 == 0, with_indices=True)

# In[46]:


from transformers import TrainingArguments, Trainer

output_dir="peft-lora-flan-t5-base"

# Define training args
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=2,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
)
    
# Create Trainer instance
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)
peft_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# # Train PEFT adapter

# In[48]:


peft_model_path="./peft-dialogue-summary-checkpoint"

# peft_trainer.train()
# peft_trainer.model.save_pretrained(peft_model_path, model_id=peft_model_id)
# tokenizer.save_pretrained(peft_model_path)
# !aws s3 cp --recursive ./peft-dialogue-summary-checkpoint/ s3://dsoaws/models/peft-dialogue-summary-checkpoint/

# # Read PEFT model from S3

# In[49]:


!aws s3 cp --recursive s3://dsoaws/models/peft-dialogue-summary-checkpoint/ ./peft-dialogue-summary-checkpoint/ 

# # Show size of PEFT adapter layers/parameters
# Much less than original LLM.

# In[50]:


!ls -al ./peft-dialogue-summary-checkpoint/adapter_model.bin

# In[51]:


import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# load base LLM model and tokenizer
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Load the LoRA/PEFT model
peft_model = PeftModel.from_pretrained(peft_model_base, './peft-dialogue-summary-checkpoint/', device_map="auto")
peft_model.eval()

print("Peft model loaded")

# # Qualitative results

# In[56]:


idx = 20
diag = dataset['test'][idx]['dialogue']
baseline_human_summary = dataset['test'][idx]['summary']

prompt = f'Summarize the following conversation.\n\n{diag}\n\nSummary:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

print(f'Prompt:\n--------------------------\n{prompt}\n--------------------------')
print(f'Baseline human summary from original dataset: {baseline_human_summary}')
print(f'Original Flan-T5 summary: {original_model_text_output}')
print(f'Instruct Flan-T5 summary: {instruct_model_text_output}')
print(f'PEFT model summary: {peft_model_text_output}')

# # Quantitative Results with ROGUE Metric
# 
# The [ROUGE metric](https://en.wikipedia.org/wiki/ROUGE_(metric)) helps quantify the validity of summarizations produced by models. It compares summarizations to a "baseline" summary which is usually created by a human. While not perfect, it does give an indication to the overall increase in summarization effectiveness that we have accomplished by fine-tuning.

# ## Evaluate a Subsection of Summaries

# In[57]:


# again, for the sake of time, we will only be generating 10 summarizations with each model
# outside of the lab, a good exercise is to increase the number of validation summaries generated
dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']

original_model_summaries = []
instruct_model_summaries = []
peft_model_summaries = []

for idx, diag in enumerate(dialogues):
    prompt = f'Summarize the following conversation.\n\nConversation:\n{diag}\n\nSummary:'
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # human baseline
    human_baseline_text_output = human_baseline_summaries[idx]
    
    # original model
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    # instruct model
    instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

    # peft model
    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    original_model_summaries.append(original_model_text_output)
    instruct_model_summaries.append(instruct_model_text_output)
    peft_model_summaries.append(peft_model_text_output)

# In[58]:


# import pandas as pd
 
# zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries, peft_model_summaries))
 
# df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'original_model_summaries', 'instruct_model_summaries', 'peft_model_summaries'])
# df.to_csv("diag-summary-training-results-with-peft.csv")
# df

# # Compute ROUGE score for subset of data

# In[72]:


rouge = evaluate.load('rouge')

# In[73]:


original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
original_model_results

# In[74]:


instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
instruct_model_results

# In[75]:


peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
peft_model_results

# In[86]:


flan_t5_base

# In[90]:


flan_t5_base_instruct_full

# In[88]:


flan_t5_base_instruct_peft

# ## Evalute the Full Dataset
# 
# The file called "diag-summary-training-results-with-peft.csv" contains a pre-populated list of all model results which we can use to evaluate on a larger section of data. The results show substantial improvement in all ROUGE metrics!

# In[76]:


import pandas as pd
results = pd.read_csv("diag-summary-training-results-with-peft.csv")
human_baseline_summaries = results['human_baseline_summaries'].values
original_model_summaries = results['original_model_summaries'].values
instruct_model_summaries = results['instruct_model_summaries'].values
peft_model_summaries     = results['peft_model_summaries'].values

# In[77]:


original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
original_model_results

# In[78]:


instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
instruct_model_results

# In[79]:


peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)
peft_model_results

# # Calculate improvement of PEFT over original

# In[80]:


improvement = (np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key} absolute percentage improvement of peft model over original model: {value*100:.2f}%')

# # Calculate improvement of PEFT over Instruct

# In[81]:


improvement = (np.array(list(peft_model_results.values())) - np.array(list(instruct_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key} absolute percentage improvement of peft model over instruct model: {value*100:.2f}%')

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



