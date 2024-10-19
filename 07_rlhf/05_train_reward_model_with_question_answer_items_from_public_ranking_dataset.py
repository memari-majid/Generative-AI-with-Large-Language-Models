#!/usr/bin/env python
# coding: utf-8

# # Explore the reward dataset used to improve the model's helpfulness
# 
# Dataset is from this repo: https://huggingface.co/datasets/lvwerra/stack-exchange-paired/viewer/lvwerra--stack-exchange-paired/train

# In[2]:


# %pip install torch==2.0.1 torchdata
#%pip install torch==1.13.1 torchdata

# In[3]:


# %pip install --disable-pip-version-check -q \
#     transformers==4.34.1 \
#     datasets==2.12.0 \
#     accelerate==0.23.0 \
#     evaluate==0.4.0 \
#     trl==0.7.2 \
#     rouge_score==0.1.2 \
#     loralib==0.1.1

# In[6]:


# reward_modeling.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    model_name: Optional[str] = field(        
        #default="gpt2",
        #default="EleutherAI/gpt-neo-125m",
        default="roberta-base",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=50000,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "Enables gradient checkpointing."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(
        default=512
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
print(script_args)

# In[7]:


# Load the human stack-exchange-paired dataset for training the reward model.
train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/reward", split="train")
if script_args.train_subset > 0:
    train_dataset = train_dataset.select(range(script_args.train_subset))

validation_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train")
if script_args.eval_subset > 0:
    validation_dataset = validation_dataset.select(range(script_args.eval_subset))

# In[8]:


model_name_split = script_args.model_name.split("/")[-1]
# output_name = (
#     f"{model_name_split}_peft_stack-exchange-paired_rmts__{script_args.train_subset}_{script_args.learning_rate}"
# )


# Load the value-head model and tokenizer.
config = AutoConfig.from_pretrained(script_args.model_name)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8, # rank
    lora_alpha=32,
    lora_dropout=0.1,
)

peft_ranking_reward_public_qanda_model_base = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16
)
peft_ranking_reward_public_qanda_model = get_peft_model(peft_ranking_reward_public_qanda_model_base, peft_config)
peft_ranking_reward_public_qanda_model.print_trainable_parameters()
#peft_rl_ranking_reward_public_dataset_model.config.pad_token_id = tokenizer.eos_token_id # needed for gpt2, gpt-neo, etc
#peft_rl_ranking_reward_public_dataset_model.config.use_cache = not script_args.gradient_checkpointing

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
# tokenizer.pad_token = tokenizer.eos_token # needed for gpt2, gpt-neo

num_proc = 24  # Can adjust to be higher if you have more processors.

# In[9]:


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples

# In[10]:


original_columns = train_dataset.column_names

# preprocess the dataset and filter out QAs that are longer than script_args.max_length
train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
train_dataset = train_dataset.filter(lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length)

# In[11]:


validation_dataset = validation_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
validation_dataset = validation_dataset.filter(lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length)

# In[12]:


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
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
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
        
        # print(batch)
        
        return batch

# In[13]:


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    #print('predictions {}'.format(predictions))
    labels = np.zeros(predictions.shape)
    #print('labels {}'.format(labels))
    metrics = accuracy.compute(predictions=predictions, references=labels)
    #print('metrics {}'.format(metrics))
    return metrics


class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        predicted_rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        print('shape rewards_j: {}'.format(predicted_rewards_j.shape))        
        
        predicted_rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        print('shape rewards_k: {}'.format(predicted_rewards_k.shape))
        
        loss = -nn.functional.logsigmoid(predicted_rewards_j - predicted_rewards_k).mean()
        
        print('return_outputs {}'.format({"loss": loss, "rewards_j": predicted_rewards_j, "rewards_k": predicted_rewards_k}))
        if return_outputs:
            return loss, {"rewards_j": predicted_rewards_j, "rewards_k": predicted_rewards_k}
        return loss


# In[14]:


peft_ranking_reward_public_qanda_checkpoint='./peft_ranking_reward_public_qanda/'
    
# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
training_args = TrainingArguments(
    output_dir=peft_ranking_reward_public_qanda_checkpoint,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,

#   evaluation_strategy="steps",
#   eval_steps=500,
#   save_strategy="steps",
#   save_steps=500,
    
#    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
#    gradient_checkpointing=script_args.gradient_checkpointing,
#    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=1, # was 1000
    max_steps=10, # was 1000
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
)

# Train the reward model, finally!
reward_trainer = RewardTrainer(
    model=peft_ranking_reward_public_qanda_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
)

# In[15]:


reward_trainer.train(script_args.resume_from_checkpoint)

# In[16]:


print("Saving last checkpoint of the model to {}".format(peft_ranking_reward_public_qanda_checkpoint))
#peft_rl_ranking_reward_public_dataset_model.save_pretrained(peft_rl_ranking_reward_public_dataset_model_checkpoint_name)
#reward_trainer.tokenizer.save_pretrained(peft_rl_ranking_reward_public_dataset_model)
#reward_trainer.unwrap_model(reward_trainer.model).save_pretrained(peft_rl_ranking_reward_public_dataset_model) # merge?
#reward_trainer.model.save_pretrained(peft_rl_ranking_reward_public_dataset_model_checkpoint_name)

merged_model = peft_ranking_reward_public_qanda_model.merge_and_unload()
merged_model.save_pretrained(peft_ranking_reward_public_qanda_checkpoint)
tokenizer.save_pretrained(peft_ranking_reward_public_qanda_checkpoint)  #    output_name + "rl_reward_model")??

# In[17]:


%store peft_ranking_reward_public_qanda_checkpoint

# In[18]:


peft_ranking_reward_public_qanda_model = AutoModelForSequenceClassification.from_pretrained(peft_ranking_reward_public_qanda_checkpoint, num_labels=1)

# In[19]:


from transformers import TextClassificationPipeline
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained(peft_ranking_reward_public_qanda_checkpoint)

peft_ranking_reward_public_qanda_pipeline = pipeline("text-classification", tokenizer=tokenizer, model=peft_ranking_reward_public_qanda_checkpoint)

# In[20]:


question = 'Who was not the President of the United States in 2010?'
answer = 'Barack Obama'
prompt_and_answer = "Question: " + question + "\n\nAnswer: " + answer + "\n"
peft_ranking_reward_public_qanda_pipeline.predict(prompt_and_answer)

# In[ ]:



