#!/usr/bin/env python
# coding: utf-8

# # Train reward model with human feedback
# 
# The reward model is trained on a human-labeled dataset with the preferred `star_rating` for a given review.  The model flattens the human-labeled data from Ground Truth into (review, star_rating, ranking) tuples and provides a reward score for the RL-based model fine-tuning.
# 
# ![Pipeline](img/generative_ai_pipeline_rlhf_plus.png)
# 
# ![RLHF](img/rlhf_qa.png)
# 
# ![Convert human ranking data into reward dataset](img/convert_groundtruth_ranking_data_to_reward_model_dataset_qa.png)

# In[2]:


# question1 response1 response2    0 
# question1 response1 response3    1
# question1 response1 response4    0

# question1 response2 response3    0 
# question1 response2 response4    1

# question1 response3 response4    0

# In[8]:


# %pip install --disable-pip-version-check -q \
#     transformers==4.26.1 \
#     datasets==2.9.0 \
#     accelerate==0.17.0 \
#     bitsandbytes==0.37.0 \
#     promptsource==0.2.3 \
#     trl==0.4.1 \
#     evaluate==0.4.0

# In[18]:


import boto3
import sagemaker
import pandas as pd

sess = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

# In[19]:


import io
import json
import uuid
import time
import boto3
import botocore

# Amazon Python SDK clients
sagemaker = boto3.client("sagemaker", region)
a2i = boto3.client("sagemaker-a2i-runtime")
s3 = boto3.client("s3", region)

# In[20]:


import os
import glob
import numpy as np
import argparse
import pprint
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

# In[21]:


%store -r human_feedback_dataset

# In[22]:


try:
    human_feedback_dataset
except NameError:
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("[ERROR] Please run the notebooks in the previous section before you continue.")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# In[23]:


print(human_feedback_dataset)

# # Train a reward model with human preference and alignment data
# This is typically a language model initialized from the supervised-fine-tuned (SFT) model (trained in a previous notebook), but with an additional binary-classification layer placed on top.  This reward model is used to train the reinforcement-learning model in the next step.  The reinforcement-learning model is what is deployed into production to serve applications.

# In[30]:


%store -r peft_fine_tuned_with_public_qanda

# In[31]:


try:
    peft_fine_tuned_with_public_qanda
except NameError:
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("[ERROR] Please run the notebooks in the previous section before you continue.")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# In[32]:


print(peft_fine_tuned_with_public_qanda)

# In[33]:


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

# In[40]:


from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

base_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small',
                                                   torch_dtype=torch.float16)

model = PeftModel.from_pretrained(base_model, peft_fine_tuned_with_public_qanda)
tokenizer = AutoTokenizer.from_pretrained(peft_fine_tuned_with_public_qanda)

# In[41]:


# Turn the dataset into pairs of prompt + responses, where text_j is the preferred prompt + response and text_k is the other.
def turn_into_text_classification_format(examples):
    new_examples = {"text_j": [], "text_k": []}
    print(new_examples)
    for prompt, response, ranking in zip(examples["prompt"], examples["response"], examples["ranking"]):
        # TODO:  Add a check to make sure there is only a single 0 and a single 1
        if len(response) != 2 or len(ranking) != 2 or ranking[0] not in (0, 1) or ranking[1] not in (0, 1):
            raise ValueError(
                f"There should be two responses with a ranking that is either 0 or 1. Received {len(response)} responses and {len(ranking)} rankings."
            )
            
        highest_ranked_response_index = ranking.index(1) # find the response (from a list of 2 responses) 
                                                         # with reward 1 as defined by the human labeler

        new_examples["text_j"].append(
            #str(response[highest_ranked_response_index]) + " " + tokenizer.bos_token + " " + prompt
            prompt + " " + str(response[highest_ranked_response_index])
        )
        new_examples["text_k"].append(
            #str(response[0 if highest_ranked_response_index == 1 else 1]) + " " + tokenizer.bos_token + " " + prompt
            prompt + " " + str(response[0 if highest_ranked_response_index == 1 else 1])
        )

    return new_examples

# Tokenize the dataset.
def preprocess_function(examples):
    tokenized_j = tokenizer(examples["text_j"], truncation=True)
    tokenized_k = tokenizer(examples["text_k"], truncation=True)
    return {
        "input_ids_j": tokenized_j["input_ids"],
        "attention_mask_j": tokenized_j["attention_mask"],
        "input_ids_k": tokenized_k["input_ids"],
        "attention_mask_k": tokenized_k["attention_mask"],
    }


# In[42]:


num_proc = 8  # Can adjust to be higher if you have more processors. Should work even if you don't have 8 CPUs, though.
original_columns = human_feedback_dataset.column_names
print(original_columns)

human_feedback_binary_classification_dataset = human_feedback_dataset.map(turn_into_text_classification_format, batched=True, num_proc=num_proc, remove_columns=original_columns)

human_feedback_tokenized_dataset = human_feedback_binary_classification_dataset.map(preprocess_function, 
                                                                                    batched=True, 
                                                                                    num_proc=num_proc, 
                                                                                    remove_columns=["text_j", "text_k"])

print(human_feedback_tokenized_dataset)


# In[43]:


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

# In[44]:


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append({"input_ids": feature["input_ids_j"], "attention_mask": feature["attention_mask_j"]})
            features_k.append({"input_ids": feature["input_ids_k"], "attention_mask": feature["attention_mask_k"]})
        batch_j = self.tokenizer.pad( # question answer pair
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch

# In[45]:


peft_ranking_reward_custom_qanda_model_name = 'roberta-base'
peft_ranking_reward_custom_qanda_model = AutoModelForSequenceClassification.from_pretrained(peft_ranking_reward_custom_qanda_model_name, num_labels=1)

# In[46]:


class RewardTrainer(Trainer):
    # Define how to compute the reward loss.
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

# Define and parse arguments.
local_rank = 0
resume_from_checkpoint = False
deepspeed = None
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
gradient_accumulation_steps = 4
learning_rate = 2e-5
weight_decay = 0.001
bf16 = False
num_train_epochs = 1

peft_ranking_reward_custom_qanda_checkpoint = './peft_ranking_reward_model_custom_qanda/'

# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
training_args = TrainingArguments(
    output_dir=peft_ranking_reward_custom_qanda_checkpoint,
    learning_rate=learning_rate,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
#    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=gradient_accumulation_steps,
#    deepspeed=deepspeed,
#    local_rank=local_rank,
    remove_unused_columns=False,
    label_names=[],
)
    
# Train the model, woohoo.
trainer = RewardTrainer(
    model=peft_ranking_reward_custom_qanda_checkpoint,
    args=training_args,
    train_dataset=human_feedback_tokenized_dataset, #["train"],
#    eval_dataset=tokenized_ds["validation"],
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train(resume_from_checkpoint)

# In[47]:


trainer.save_model(peft_ranking_reward_custom_qanda_checkpoint)
tokenizer.save_pretrained(peft_ranking_reward_custom_qanda_checkpoint)

# In[48]:


%store peft_ranking_reward_custom_qanda_checkpoint

# In[49]:


peft_ranking_reward_custom_qanda_checkpoint = AutoModelForSequenceClassification.from_pretrained(peft_ranking_reward_custom_qanda_checkpoint, num_labels=1)

# In[50]:


from transformers import TextClassificationPipeline
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained(peft_ranking_reward_custom_qanda_checkpoint)

peft_ranking_reward_custom_qanda_pipeline = pipeline("text-classification", tokenizer=tokenizer, model=peft_ranking_reward_custom_qanda_checkpoint)

# In[51]:


question = 'Who was not the President of the United States in 2010?'
answer = 'Barack Obama'
prompt_and_answer = "Question: " + question + "\n\nAnswer: " + answer + "\n"
peft_ranking_reward_custom_qanda_pipeline.predict(prompt_and_answer)

# In[ ]:



