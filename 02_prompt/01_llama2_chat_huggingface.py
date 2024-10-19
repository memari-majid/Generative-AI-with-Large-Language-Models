#!/usr/bin/env python
# coding: utf-8

# In[7]:


# %pip install -U torch==2.0.1 \
#   transformers==4.33.0 \
#   sentencepiece==0.1.99 \
#   accelerate==0.22.0 # needed for low_cpu_mem_usage parameter

# In[3]:


import torch
from transformers import LlamaTokenizer

model_checkpoint = "NousResearch/Llama-2-7b-hf"
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
system_message["content"] = "Answer only with emojis"
print(system_message)

user_message = Message()
user_message["role"] = "user"
user_message["content"] = "Who won the 2016 baseball World Series?"
print(user_message)

# assistant_message = Message()
# assistant_message.role = "assistant"
# assistant_message.content = ""

list_of_messages = list()
list_of_messages.append(system_message)
list_of_messages.append(user_message)

list_of_message_lists = list()
list_of_message_lists.append(list_of_messages)

prompt = convert_list_of_message_lists_to_input_prompt(list_of_message_lists, tokenizer)
print(prompt)

# In[17]:


from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    model_checkpoint,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# model = model.eval()

# In[18]:


from transformers import pipeline

tokenized_prompt = tokenizer(prompt)

print(f'prompt is {len(tokenized_prompt["input_ids"][0])} tokens')

# In[19]:


from transformers import GenerationConfig

generation_config = GenerationConfig(max_new_tokens=2000)

pipeline = pipeline("text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                    generation_config=generation_config)

# In[ ]:


pipeline(prompt)
