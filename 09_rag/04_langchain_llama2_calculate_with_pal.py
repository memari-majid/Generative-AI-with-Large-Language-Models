#!/usr/bin/env python
# coding: utf-8

# # Tested on ml.m5.2xlarge

# In[2]:


# %pip install \
#     torch==2.0.1 \
#     transformers==4.34.1 \
#     langchain==0.0.309 \
#     langchain_experimental 
#    iprogress==0.4

# Load libraries: 

# In[3]:


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline

# Grab a LLM.  We use Hugging Face, and setup a simple pipeline.

# In[4]:


model_checkpoint = "NousResearch/Llama-2-7b-chat-hf"

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=2000)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

# In[ ]:


from langchain.llms.base import LLM
from langchain.llms import HuggingFacePipeline

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, temperature=0.8, max_new_tokens=2000
)

llm = HuggingFacePipeline(pipeline=pipe)

# In[ ]:



from langchain import PromptTemplate
from langchain_experimental.pal_chain.base import PALChain

pal_chain = PALChain.from_math_prompt(
    llm, 
    verbose=True,
)

pal_prompt_template = """Translate a math problem into a expression that can be executed using Python's numexpr library. 
Use the output of running this code to answer the question.

Question: ${{Question with hard calculation.}}
${{Code that prints what you need to know}}


Question: I have three apples and get given two more, how many apples do I have?
def solution():
    initial_apples = 3
    extra_apples = 2
    return initial_apples + extra_apples


Question: Jan has three times the number of pets as Marcia. Marcia has six pets, how many pets does jan have?
def solution():
    marcia_pets = 6
    jan_pets = marcia_pets * 3
    return jan_pets

Question: Jane has 10 sweets. Peter has twice the number of sweets than Jane. How many sweets does Peter have?"
"""

# In[ ]:


answer = pal_chain.run(pal_prompt_template)
print(answer)

# In[ ]:



