#!/usr/bin/env python
# coding: utf-8

# # Tested on ml.trn1.32xlarge

# This is based on this example:  https://github.com/huggingface/trl/tree/02f5c1d8cee73045c837d01d7f1577a57779b035/examples/research_projects/stack_llama_2/scripts

# In[2]:


# %pip install --disable-pip-version-check -U \
#     torch==2.0.1 \
#     transformers \
#     datasets==2.12.0 \
#     accelerate \
#     evaluate==0.4.0 \
#     trl \
#     rouge_score==0.1.2 \
#     loralib==0.1.1 \
#     typing_extensions==4.7.1 \
#     bitsandbytes==0.41.1 \
#     peft \
#     bitsandbytes \
#     optimum-neuron

# In[2]:


%pip install --disable-pip-version-check -q \
    torch==2.0.1 \
    transformers==4.34.1 \
    datasets==2.12.0 \
    accelerate==0.23.0 \
    evaluate==0.4.0 \
    trl==0.7.2 \
    rouge_score==0.1.2 \
    loralib==0.1.1 \
    typing_extensions==4.7.1 \
    bitsandbytes==0.41.1 \
    peft==0.5.0 \
    optimum-neuron

# In[3]:


%pip list

# In[4]:


# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM 

from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser 
from transformers import TrainingArguments

from optimum.neuron import NeuronTrainer #as Trainer
from optimum.neuron import NeuronTrainingArguments #as TrainingArguments

#from trl import DPOTrainer
from trl_neuron import NeuronDPOTrainer


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="NousResearch/Llama-2-7b-hf"
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=100, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="none",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

# In[5]:


def get_stack_exchange_paired(
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        "lvwerra/stack-exchange-paired",
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

# In[6]:


parser = HfArgumentParser(ScriptArguments)

script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

# 1. load a pretrained model
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
#    load_in_4bit=True,
)
model.config.use_cache = False

if script_args.ignore_bias_buffers:
    # torch distributed hack
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    ]

model_ref = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
#    load_in_4bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# 2. Load the Stack-exchange paired dataset
train_dataset = get_stack_exchange_paired(data_dir="data/rl", # https://huggingface.co/datasets/lvwerra/stack-exchange-paired/tree/main/data/rl
                                          sanity_check=False)
train_dataset = train_dataset.filter(
    lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
)

# 3. Load evaluation dataset
eval_dataset = get_stack_exchange_paired(data_dir="data/evaluation", # https://huggingface.co/datasets/lvwerra/stack-exchange-paired/tree/main/data/evaluation
                                         sanity_check=False)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
)

# 4. initialize training arguments:
training_args = NeuronTrainingArguments(
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    max_steps=script_args.max_steps,
    logging_steps=script_args.logging_steps,
    save_steps=script_args.save_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    learning_rate=script_args.learning_rate,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_steps,
    output_dir=script_args.output_dir,
    report_to=script_args.report_to,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.warmup_steps,
    optim=script_args.optimizer_type,
    bf16=True,
    remove_unused_columns=False,
    run_name="dpo_llama2",
)

peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "out_proj",
        "fc_in",
        "fc_out",
        "wte",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# TODO:  Is there a NeuronDPOTrainer equivalent?
# 5. initialize the DPO trainer
dpo_trainer = NeuronDPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=script_args.beta,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_prompt_length=script_args.max_prompt_length,
    max_length=script_args.max_length,
)

# 6. train
dpo_trainer.train()
dpo_trainer.save_model(script_args.output_dir)

# 7. save
output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
dpo_trainer.model.save_pretrained(output_dir)

# In[ ]:


from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "results/final_checkpoint",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
#    load_in_4bit=True,
)

# In[ ]:


#model.generate(...)
