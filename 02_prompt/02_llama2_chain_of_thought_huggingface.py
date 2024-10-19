#!/usr/bin/env python
# coding: utf-8

# # Tested on a g5.4xlarge

# In[2]:


# %pip install -U torch==2.0.1 \
#   transformers==4.33.0 \
#   sentencepiece==0.1.99 \
#   accelerate==0.22.0 # needed for low_cpu_mem_usage parameter

# In[3]:


import torch
from transformers import LlamaTokenizer

model_checkpoint = "NousResearch/Llama-2-13b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint)

# In[4]:


# based on https://github.com/viniciusarruda/llama-cpp-chat-completion-wrapper/blob/1c9e29b70b1aaa7133d3c7d7b59a92d840e92e6d/llama_cpp_chat_completion_wrapper.py

from typing import List
from typing import Literal
from typing import TypedDict

from transformers import PreTrainedTokenizer

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

MessageList = List[Message]

BEGIN_INST, END_INST = "[INST] ", " [/INST] "
BEGIN_SYS, END_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def convert_list_of_message_lists_to_input_prompt(list_of_message_lists: List[MessageList], tokenizer: PreTrainedTokenizer) -> List[str]:
    input_prompts: List[str] = []
    print(type(list_of_message_lists))
    print(type(list_of_message_lists[0]))    
    for message_list in list_of_message_lists:
        if message_list[0]["role"] == "system":
            content = "".join([BEGIN_SYS, message_list[0]["content"], END_SYS, message_list[1]["content"]])
            message_list = [{"role": message_list[1]["role"], "content": content}] + message_list[2:]

        if not (
            all([msg["role"] == "user" for msg in message_list[::2]])
            and all([msg["role"] == "assistant" for msg in message_list[1::2]])
        ):
            raise ValueError(
                "Format must be in this order: 'system', 'user', 'assistant' roles.\nAfter that, you can alternate between user and assistant multiple times"
            )

        eos = tokenizer.eos_token
        bos = tokenizer.bos_token
        input_prompt = "".join(
            [
                "".join([bos, BEGIN_INST, (prompt["content"]).strip(), END_INST, (answer["content"]).strip(), eos])
                for prompt, answer in zip(message_list[::2], message_list[1::2])
            ]
        )

        if message_list[-1]["role"] != "user":
            raise ValueError(f"Last message must be from user role. Instead, you sent from {message_list[-1]['role']} role")

        input_prompt += "".join([bos, BEGIN_INST, (message_list[-1]["content"]).strip(), END_INST])

        input_prompts.append(input_prompt)

    return input_prompts

# In[5]:


system_message = Message()
system_message["role"] = "system"
system_message["content"] = ""
print(system_message)

user_message = Message()
user_message["role"] = "user"
user_message["content"] = """
QUESTION: Ducks need to eat 3.5 pounds of insects each week to survive. 
If there is a flock of ten ducks, how many pounds of insects do they need per day?
ANSWER: Ducks need 3.5 pounds of insects each week. If there is a flock of 10 ducks, then they need 3.5 x 10 = 35 pounds of insects each week. If they need 35 pounds of insects each week, then they need 35 / 7 = 5 pounds of insects each day. The answer is 5. 

QUESTION: It takes Matthew 3 minutes to dig a small hole for shrubs and 10 minutes to dig a large hole for trees.
How many hours will it take him to dig 30 small holes and 15 large holes?
ANSWER: It takes Matthew 3 minutes to dig a small hole and 10 minutes to dig a large hole. So, it takes Matthew 3 x 30 = 90 minutes to dig 30 small holes. It takes Matthew 10 x 15 = 150 minutes to dig 15 large holes. So, it takes Matthew 90 + 150 = 240 minutes to dig 30 small holes and 15 large holes. 240 minutes is 4 hours. The answer is 4 hours. 

QUESTION: I have 10 liters of orange drink that are two-thirds water and I wish to add it to 15 liters of pineapple drink that is three-fifths water. 
But as I pour it, I spill one liter of the orange drink. How much water is in the remaining 24 liters?
ANSWER:

"""

list_of_messages = list()
list_of_messages.append(system_message)
list_of_messages.append(user_message)

list_of_message_lists = list()
list_of_message_lists.append(list_of_messages)

prompt = convert_list_of_message_lists_to_input_prompt(list_of_message_lists, tokenizer)
print(prompt)

# In[6]:


from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    model_checkpoint,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

model = model.eval()

# In[7]:


from transformers import pipeline

tokenized_prompt = tokenizer(prompt)

print(f'prompt is {len(tokenized_prompt["input_ids"][0])} tokens')

# In[8]:


from transformers import GenerationConfig

generation_config = GenerationConfig(max_new_tokens=2000)                                      

pipeline = pipeline("text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                    generation_config=generation_config)

# ### This next cell takes about 10-15 minutes.  Please be patient.

# In[ ]:


pipeline(prompt, return_full_text=False)

# In[ ]:



