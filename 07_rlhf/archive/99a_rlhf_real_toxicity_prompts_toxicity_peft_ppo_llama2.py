#!/usr/bin/env python
# coding: utf-8

# In[2]:


%pip install torch==2.0.1 torchdata

%pip install --disable-pip-version-check -q \
    transformers==4.34.1 \
    datasets==2.12.0 \
    accelerate==0.23.0 \
    evaluate==0.4.0 \
    trl==0.7.1 \
    rouge_score==0.1.2 \
    loralib==0.1.1

# In[3]:


!pip install git+https://github.com/huggingface/peft.git

# In[4]:


# !pip install git+https://github.com/lvwerra/trl.git

# In[5]:


from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

tqdm.pandas()

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="NousResearch/Llama-2-7b-chat-hf", metadata={"help": "the model name"})    
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./peft_fine_tuned_with_detoxification_rewards",
        metadata={"help": "the path to save the model"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

# In[6]:


from trl.core import LengthSampler

def build_dataset(
    tokenizer, dataset_name="allenai/real-toxicity-prompts", input_min_text_length=5, input_max_text_length=10
):
    ds = load_dataset(dataset_name, split="train")

    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > 0.3

    ds = ds.filter(filter_fn, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]

        sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

    return ds


# In[7]:


tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, 
                                          device_map="auto")

min_input_length = 30
max_input_length = 40

dataset = build_dataset(tokenizer, input_min_text_length=min_input_length, input_max_text_length=max_input_length)

# In[8]:


# Let's re-use Facebook/Meta's detoxification model to compute the reward.
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_id, 
                                                   device_map="auto")

# We load the model in fp16 to save memory.
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_id, 
                                                                    torch_dtype=torch.bfloat16,
                                                                    device_map="auto")
  #.to(ppo_trainer.accelerator.device)

# In[9]:


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": 4, # changed from -1 to workaround "must be 4 min tokens" error
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
#    "pad_token_id": tokenizer.eos_token_id,
}

# In[10]:


from transformers import TextClassificationPipeline
from transformers import pipeline

toxicity_pipeline = pipeline("text-classification", 
                             tokenizer=toxicity_tokenizer, 
                             model=toxicity_model)

# In[11]:


super_toxic_dataset = dataset.filter(lambda row: row["prompt"]["identity_attack"] > 0.80)
super_toxic_dataset

# In[12]:


print(super_toxic_dataset[200])

toxicity_pipeline.predict(super_toxic_dataset[0]["prompt"]["text"])

# In[13]:


from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed

from peft import LoraConfig

# We load the model in bf16 to save memory.
model = AutoModelForCausalLM.from_pretrained(script_args.model_name, 
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(model, 
                                                          peft_config=lora_config,
                                                          device_map="auto")

ref_model = create_reference_model(model)

# In[14]:


config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=1, # was 100
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)

# In[15]:


# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# In[ ]:


output_min_length = 20
output_max_length = 200
output_length_sampler = LengthSampler(output_min_length, output_max_length)

peft_fine_tuned_with_detoxification_rewards_checkpoint = script_args.model_save_path

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from the policy model
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute toxicity score for the response pair
    texts = batch["response"]
    toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        ppo_trainer.accelerator.device
    )
    logits = toxicity_model(**toxicity_inputs).logits.float()
    toxicity_labels = (logits[:, 0]).tolist()

    rewards = [torch.tensor(output) for output in toxicity_labels]

    # Run PPO gradient-update step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    if epoch % 100 == 0:
        if ppo_trainer.accelerator.is_main_process:
            #ppo_trainer.save_pretrained(peft_fine_tuned_with_detoxification_rewards_checkpoint) # depends on huggingface hub
            ppo_trainer.tokenizer.save_pretrained(peft_fine_tuned_with_detoxification_rewards_checkpoint)
            ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained(peft_fine_tuned_with_detoxification_rewards_checkpoint) # merge
            #ppo_trainer.model.save_pretrained(peft_fine_tuned_with_detoxification_rewards_checkpoint)

# # Save model

# In[ ]:


#ppo_trainer.save_pretrained(model_save_path) # depends on huggingface hub
ppo_trainer.tokenizer.save_pretrained(peft_fine_tuned_with_detoxification_rewards_checkpoint)
#ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained(peft_fine_tuned_with_detoxification_rewards_checkpoint) # merge?
ppo_trainer.model.save_pretrained(peft_fine_tuned_with_detoxification_rewards_checkpoint)

# In[ ]:


# %store peft_fine_tuned_with_detoxification_rewards_checkpoint

# In[ ]:


from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

reward_model = AutoModelForCausalLM.from_pretrained(peft_fine_tuned_with_detoxification_rewards_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(peft_fine_tuned_with_detoxification_rewards_checkpoint)

# In[ ]:




# In[ ]:



