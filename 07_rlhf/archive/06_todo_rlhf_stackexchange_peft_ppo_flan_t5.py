#!/usr/bin/env python
# coding: utf-8

# # Tested on ml.m5.4xlarge

# In[2]:


%pip install torch==2.0.1 torchdata

# In[3]:


%pip install --disable-pip-version-check -q \
    transformers==4.34.1 \
    datasets==2.12.0 \
    accelerate==0.23.0 \
    evaluate==0.4.0 \
    trl==0.7.2 \
    rouge_score==0.1.2 \
    loralib==0.1.1 \
    typing_extensions==4.7.1 \
    peft

# In[4]:


%store -r peft_ranking_reward_public_qanda_checkpoint

# In[5]:


print(peft_ranking_reward_public_qanda_checkpoint)

# In[6]:


from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


tqdm.pandas()

peft_fine_tuned_with_ranking_rewards_flan_t5_checkpoint='./peft_fine_tuned_with_ranking_rewards_flan_t5'

@dataclass
class ScriptArguments:    
    model_name: Optional[str] = field(default="google/flan-t5-base", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="google/flan-t5-base", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default=peft_ranking_reward_public_qanda_checkpoint, metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.4e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=True, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=True, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=100, metadata={"help": "n steps to save the model"})
#    output_dir: Optional[str] = field(default=fine_tuned_with_ranking_rewards_checkpoint, metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
dataset_name = "lvwerra/stack-exchange-paired"

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
)


# Dataset is here:
#   https://huggingface.co/datasets/lvwerra/stack-exchange-paired/tree/main/data/rl

train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
train_dataset = train_dataset.select(range(100000))
# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16, "truncation": True}

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)

# Below is an example function to build the dataset. In our case, we use the StackExchange Q&A dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer, dataset_name="lvwerra/stack-exchange-paired", input_min_text_length=2, input_max_text_length=8
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # load with datasets
    ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
    original_columns = ds.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
#current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    config.model_name,
#    load_in_8bit=True,
#    device_map={"": current_device},
    peft_config=lora_config,
#    layer_norm_names=[],
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None, # TODO: add ref mode for kl-divergence
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
# device = ppo_trainer.accelerator.device
# if ppo_trainer.accelerator.num_processes == 1:
#     device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug


reward_model_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_model_name)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=script_args.reward_model_name, #roberta reward model
#    device_map={"": current_device},
#    model_kwargs={"load_in_8bit": True},
    tokenizer=reward_model_tokenizer,
)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    prompt_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        prompt_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs) # roberta reward prediction between 0 and 1
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

    # Run PPO step - perform gradient update on the flan-t5 model
    stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        #ppo_trainer.save_pretrained(script_args.output_dir) # depends on huggingface hub
        ppo_trainer.tokenizer.save_pretrained(peft_fine_tuned_with_ranking_rewards_flan_t5_checkpoint)
        ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained(peft_fine_tuned_with_ranking_rewards_flan_t5_checkpoint) # merge
        #ppo_trainer.model.save_pretrained(peft_fine_tuned_with_detoxification_rewards_checkpoint)

# In[ ]:


print(peft_fine_tuned_with_ranking_rewards_flan_t5_checkpoint)

# In[ ]:


ppo_trainer.tokenizer.save_pretrained(peft_fine_tuned_with_ranking_rewards_flan_t5_checkpoint)
ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained(peft_fine_tuned_with_ranking_rewards_flan_t5_checkpoint) # merge?

# In[ ]:


%store peft_fine_tuned_with_ranking_rewards_flan_t5_checkpoint

# In[ ]:



