#!/usr/bin/env python
# coding: utf-8

# # Bedrock Prompt Examples for Self-Evaluation
# 
# In this notebook, we include examples for having the models self-evaluating their responses through Prompt Evaluation.

# Let's start by ensuring the Bedrock SDK is properly installed.
# 
# We'll also install a few libraries required in the notebook.

# In[ ]:


#!unzip ../bedrock-preview-documentation/SDK/bedrock-python-sdk.zip -d /root/bedrock

#!pip install --upgrade pip
#!pip install scikit-learn seaborn

#!pwd
#!python3 -m pip install /root/bedrock/boto3-1.26.142-py3-none-any.whl
#!python3 -m pip install /root/bedrock/botocore-1.29.142-py3-none-any.whl

# Now we can import the libraries and setup the Bedrock client.

# In[1]:


import boto3
import json
import csv
from datetime import datetime

bedrock = boto3.client(
 service_name='bedrock',
 region_name='us-east-1',
 endpoint_url='https://bedrock.us-east-1.amazonaws.com'
)

# Let's get the list of Foundational Models supported in Bedrock at this time.

# In[2]:


bedrock.list_foundation_models()

# We will define an utility function for calling Bedrock.
# 
# This will help passing the proper body depending on the model invoked, and will store the results in a CSV file as well.

# In[26]:


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
        body = json.dumps({
            "prompt": prompt_data,
            "max_tokens_to_sample": 8000,
            "stop_sequences":[],
            "temperature":0.2,
            "top_p":0.9
        })
        #modelId = 'anthropic.claude-instant-v1'
    elif 'ai21' in modelId:
        body = json.dumps({
            "prompt": prompt_data,
            "maxTokens":4096,
            "stopSequences":[],
            "temperature":0,
            "topP":0.9
        })
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
    #data = [datetime.now(), modelId, prompt_data, response, latency] #the data
    #with open('./prompt-data/prompt-data.csv', 'a') as f:
    #    writer = csv.writer(f)
    #    #writer.writerow(column_name)
    #    writer.writerow(data)
    
    return response, latency

# Now we are ready for running our examples with different models.
# 
# -----

# ## 1. Prompt Self-Evaluation

# ### Titan

# In[95]:


prompt_data ="""
Context: The 1881 world tour of King Kalākaua of the Hawaiian Kingdom was his attempt to save the Hawaiian culture and population from extinction by importing a labor force from Asia-Pacific nations. His efforts brought the small island nation to the attention of world leaders, but sparked rumors that the kingdom was for sale. Critics in Hawaii believed the labor negotiations were just an excuse to see the world. The 281-day trip gave Kalākaua the distinction of being the first monarch to circumnavigate the globe; his 1874 travels had made him the first reigning monarch to visit the United States and the first honoree of a state dinner at the White House.  

Question 1: What rumors were sparked following King Kalākaua's world tour? 
Answer 1:
"""

# In[96]:


response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[97]:


prompt_data = prompt_data + response + """

Question 2: Is the 'Answer 1' Mostly Correct, Partially Correct, Partially Incorrect, or Mostly Incorrect ?
Answer 2:
"""

# In[98]:


response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ### Claude

# In[29]:


prompt_data ="""
Human: Write an essay on the macro economic situation. Answer with more than 1000 words. Count the total number of words in the response after answering.
Answer:
"""

# In[30]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[47]:


prompt_data = prompt_data + response + """

Human: Considering the Context, Question, and Answer above, classify the Answer according to its accuracy choosing from 'Mostly Correct', 'Partially Correct', 'Partially Incorrect', 'Mostly Incorrect'.

Assistant:
"""

# In[48]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ### Jurassic-2

# In[61]:


prompt_data ="""
Context: The 1881 world tour of King Kalākaua of the Hawaiian Kingdom was his attempt to save the Hawaiian culture and population from extinction by importing a labor force from Asia-Pacific nations. His efforts brought the small island nation to the attention of world leaders, but sparked rumors that the kingdom was for sale. Critics in Hawaii believed the labor negotiations were just an excuse to see the world. The 281-day trip gave Kalākaua the distinction of being the first monarch to circumnavigate the globe; his 1874 travels had made him the first reigning monarch to visit the United States and the first honoree of a state dinner at the White House.  

Question: What rumors were sparked following King Kalākaua's world tour? 
Answer:
"""

# In[62]:


response, latency = call_bedrock('ai21.j2-grande-instruct', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[63]:


prompt_data = prompt_data + response + """

Considering the Context, Question, and Answer above, classify the Answer according to its accuracy choosing from 'Mostly Correct', 'Partially Correct', 'Partially Incorrect', 'Mostly Incorrect'.

"""

# In[64]:


response, latency = call_bedrock('ai21.j2-grande-instruct', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------
