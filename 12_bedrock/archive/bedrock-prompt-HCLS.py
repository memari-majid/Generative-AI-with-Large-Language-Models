#!/usr/bin/env python
# coding: utf-8

# # Introduction to Bedrock - Text Summarization Sample - Health Care and Life Sciences 

# --- 
# 
# In this demo notebook, we demonstrate how to use the Bedrock Python SDK for a text question and answer (Q&A) example. We show how to use Bedrock's Foundational Models to answer questions after reading a passage
# 
# ---

# ## 1. Set Up and API walkthrough

# ---
# Before executing the notebook for the first time, execute this cell to add bedrock extensions to the Python boto3 SDK
# 
# ---

# In[2]:


import boto3
import botocore
import pandas as pd

try:
    from langchain.chains import LLMChain
    from langchain.llms.bedrock import Bedrock
    from langchain.prompts import PromptTemplate
    from langchain.embeddings import BedrockEmbeddings
except Exception as e:
    print(e)
    print("please install langchain version 0.0.190")

# In[3]:


assert boto3.__version__ == "1.26.142"
assert botocore.__version__ == "1.29.142"

# #### Un comment these to run from your local environment outside of AWS

# In[4]:


import sys
import os
from pprint import pprint

# #### Now let's set up our connection to the Amazon Bedrock SDK using Boto3

# In[5]:


import boto3
import json

bedrock = boto3.client(
    service_name="bedrock",
    region_name="us-east-1",
    endpoint_url="https://bedrock.us-east-1.amazonaws.com",
)

# #### We can validate our connection by testing out the _list_foundation_models()_ method, which will tell us all the models available for us to use 

# In[6]:


bedrock.list_foundation_models()

# #### In this Notebook we will be using the invoke_model() method of Amazon Bedrock. This will be the primary method we use for most of our Text Generation and Processing tasks. 

# ##### The mandatory parameters required to use this method are, where _modelId_ represents the Amazon Bedrock model ARN, and _body_ which is the prompt for our task. The _body_ prompt will change depending on the foundational model provider selected. We walk through this in detail below
# 
# ```
# {
#    modelId= model_id,
#    contentType= "application/json",
#    accept= "application/json",
#    body=body
# }
# 
# ```

# ## 2. Reading the data
# We take data from the [pubmed database of scientific papers from huggingface.](https://huggingface.co/datasets/scientific_papers/viewer/pubmed/train?row=0)
# 

# In[7]:


pubmed_dataset =pd.read_csv("./huggingface_medpub_dataset_short.csv")

# In[8]:


pubmed_dataset

# In[95]:


pubmed_dataset["article_length_chars"] = pubmed_dataset.apply(lambda x: len(x['article']), axis=1)
pubmed_dataset["abstract_length_chars"] = pubmed_dataset.apply(lambda x: len(x['abstract']), axis=1)
pubmed_dataset["article_length_words"] = pubmed_dataset["article"].apply(lambda x: len(x.split(" ")))
pubmed_dataset["abstract_length_words"] = pubmed_dataset["abstract"].apply(lambda x: len(x.split(" ")))
pubmed_dataset.head()

# #### Let's now try out the Amazon Bedrock models to have it summarize our sample

# ## First try: all models, no specialised instruction set for the models

# In[96]:


models = {"titan":"amazon.titan-tg1-large", "j2":"ai21.j2-jumbo-instruct", "claude":"anthropic.claude-v1"}
example_text = pubmed_dataset["article"].iloc[3]

# In[97]:


import boto3
import json
import csv
from datetime import datetime

def call_bedrock(modelId, prompt_data):
    if 'amazon' in modelId:
        body = json.dumps({
            "inputText": prompt_data,
            "textGenerationConfig":
            {
                "maxTokenCount":4096,
                "stopSequences":[],
                "temperature":0,
                "topP":0.9
            }
        })
        #modelId = 'amazon.titan-tg1-large'
    elif 'anthropic' in modelId:
        body = json.dumps({"prompt": prompt_data, "max_tokens_to_sample": 512})
        #modelId = 'anthropic.claude-instant-v1'
    elif 'ai21' in modelId:
        body = json.dumps({"prompt": prompt_data,
                           "maxTokens":4096})
        #modelId = 'ai21.j2-grande-instruct'
    elif 'stability' in modelId:
        body = json.dumps({"text_prompts":[{"text":prompt_data}]}) 
        #modelId = 'stability.stable-diffusion-xl'
    else:
        print('Parameter model must be one of titan, claude, j2, or sd')
        return
    accept = 'application/json'
    contentType = 'application/json'

    before = datetime.now()
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    latency = (datetime.now() - before)
    response_body = json.loads(response.get('body').read())

    if 'amazon' in modelId:
        response = response_body.get('results')[0].get('outputText')
    elif 'anthropic' in modelId:
        response = response_body.get('completion')
    elif 'ai21' in modelId:
        response = response_body.get('completions')[0].get('data').get('text')

    #Add interaction to the local CSV file...
    #column_name = ["timestamp", "modelId", "prompt", "response", "latency"] #The name of the columns
    data = [datetime.now(), modelId, prompt_data, response, latency] #the data
    with open('./prompt-data/prompt-data.csv', 'a') as f:
        writer = csv.writer(f)
        #writer.writerow(column_name)
        writer.writerow(data)
    
    return response, latency

# In[98]:


example_text

# In[99]:


prompt_data =f"""Summarize the following text in 100 words or less:
{example_text}
"""

# In[100]:


response, latency = call_bedrock(models["claude"], prompt_data)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# In[101]:


response, latency = call_bedrock(models["j2"], prompt_data)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# In[102]:


response, latency = call_bedrock(models["titan"], prompt_data)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# In[103]:


# how long was our example text?
len(prompt_data.split(" "))

# # Summarization
# Adding that the prompt should be in 5 bullet points and that the style should be appropriate for a high school student. 

# ### Anthropic Claude Summarisation Deep Dive
# For Antropic Claude the order of the prompt should roughly be the following:
# - “/n/n Human:“
# - Give task context
# - Give detailed task description, including rules  and exceptions. Explain it as you would to a new employee with no context. 
# - Give examples. Inputs and outputs (optional, but improves formatting and accuracy)
# - Demonstrate the output formatting you want; deal with clauses chattiness 
# - Assistant: 
# 
# —> Assistant: My Answer is (
# 

# In[106]:


claude_prompt = f"""\n\nHuman: 
Summarize the following text in 5 sentences as bullet points. Style should be appropriate for a high-school student. 
<text>{example_text}</text>

Assistant: My Answer is (
"""

# In[105]:


response, latency = call_bedrock(models["claude"], claude_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# #### Claude: Adding \<context\> and  Switching instruction order 

# In[108]:


claude_prompt = f"""\n\nHuman: 
<context>Medical document, describing genetics defect and its symptoms.</context>
<text>{example_text}</text>

<instruction>Summarize the following text in 5 sentences as bullet points. Style should be appropriate for a high-school student.</instruction>
Assistant: My Answer is 
"""
response, latency = call_bedrock(models["claude"], claude_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# In[109]:


claude_prompt = f"""\n\nHuman: 
<instruction>Summarize the following text in 5 sentences as bullet points. Style should be appropriate for a high-school student.</instruction>
<context>Medical document, describing genetics defect and its symptoms.</context>
<text>{example_text}</text>
Assistant: My Answer is
"""
response, latency = call_bedrock(models["claude"], claude_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# ### Learnings Claude
# - Start with \n\nHuman:
# - Wrap context/input in xml tags. 
# - Context helps the model follow the requested level of depth for explanations, otherwise it will follow too much the medical speech. 
# - reversing the order does not change too much the performance. This might be if you have shorter input lenght
# 

# ## Titan
# Now we are going to look at titan model.  

# In[110]:


titan_prompt = f"""Summarize the text below in 5 sentences as bullet points. Style should be appropriate for a high-school student.
Text: {example_text}

The text is about: 
"""
response, latency = call_bedrock(models["titan"], titan_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# In[111]:


titan_prompt = f"""
{example_text}

Instruction: Summarize the text above in 5 sentences as bullet points. Style should be appropriate for a high-school student.
The text is about: 
"""
response, latency = call_bedrock(models["titan"], titan_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# ### Learning Titan:
# #### regular order: 
# Start the instruction directly without a new line on the instruction. Reference the text with a "Text:" helps. Reference the task body with an output indictor. 
# #### reversed order: 
# It does respond well to the different prompts. Adding a new line on the top, helps it to separate the example text and instructions.
# 
# A output indicator must be given, ideally referencing the task. This output indicator should directly follow the task if the order is <text><instruction><output indicator> 

# ### AI21 

# In[112]:


j2_prompt = f"""Instruction:
Summarize the text below in 5 sentences as bullet points. Style should be appropriate for a high-school student.
{example_text}

The text is about:
"""
response, latency = call_bedrock(models["j2"], j2_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# In[113]:


j2_prompt = f"""{example_text}
Instruction:
Summarize the text above in 5 sentences as bullet points. Style should be appropriate for a high-school student.

The text is about:
"""
response, latency = call_bedrock(models["j2"], j2_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# ### Learning J2:
# Adding a tag "Instruction:" helps it to follow what you ask it. 
# "Context:" does not seem to help. 
# #### regular order: 
# Does not work consistently. To be abandoned
# #### reversed order: 
# Directly start with the text to summarize. No new line, no indicator, that it is text. 
# <text><instruction><output indicator> 

# # Question and Answering
# For Q&A on the medical dataset we are going to use the [MedQuA](https://github.com/abachaa/MedQuAD) dataset. We will pick different Q&A pairs to evaluate the evoltion of our prompt. 

# In[9]:


f = open("./painkiller_leaflet_short.txt", "r")
instapainrelief_text = f.read()


# ### Claude model
# 

# In[115]:


claude_prompt = f"""\n\nHuman: 
Given the text below, answer the following questions as found in the text. Answer like you would be talking to a five year old. 
<text>{instapainrelief_text}</text>

<question>Can I drink Alcohol while taking the medicine?</question>

Assistant: My Answer is (
"""

response, latency = call_bedrock(models["claude"], claude_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# In[116]:


claude_prompt = f"""\n\nHuman: 
Given the text below, answer the following questions as found in the text. Answer like you would be talking to a five year old. If you do not know the answer, respond with "I don't know".
<text>{instapainrelief_text}</text>

<question>In what package size is InstaPainRelief available?</question>

Assistant: My Answer is (
"""

response, latency = call_bedrock(models["claude"], claude_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# ### Titan model

# In[117]:


titan_prompt = f"""{instapainrelief_text_short}

Given the text above, answer the following questions as found in the text. Answer like you would be talking to a five year old. 

Question: Can I drink Alcohol while taking the medicine?
Answer: 
"""

response, latency = call_bedrock(models["titan"], titan_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# ### J2 model

# In[118]:


j2_prompt = f"""{instapainrelief_text_short}
Given the text above, answer the following questions as found in the text. Answer like you would be talking to a five year old. 

Question: Can I drink Alcohol while taking the medicine?
Answer: 
"""
response, latency = call_bedrock(models["j2"], j2_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# # Text Generation - Chemical reaction 

# ### Claude 

# In[119]:


claude_prompt = f"""\n\nHuman: 
Sulfuric acid reacts with sodium chloride, and gives <chemical1>_____</chemical1> and <chemical2>_____<chemical2>:

Assistant: the chemical1 and chemical 2 are:
"""

response, latency = call_bedrock(models["claude"], claude_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# ### Titan

# In[120]:


titan_prompt =f"""Question:
Sulfuric acid reacts with sodium chloride, and gives _____ and _____:

Answer: the chemical compounds are:
"""

response, latency = call_bedrock(models["titan"], titan_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# ### J2 model

# In[123]:


j2_prompt = f"""Question:
Sulfuric acid reacts with sodium chloride, and gives _____ and _____:

Answer: the chemical compounds are:
"""
response, latency = call_bedrock(models["j2"], j2_prompt)
print(response, "\n\n", "Inference time:", latency, "\n\n", "Number of words:", len(response.split(" ")))

# 
