#!/usr/bin/env python
# coding: utf-8

# # Introduction to Bedrock - Building with Bedrock Embeddings

# In[ ]:


#!unzip ../bedrock-preview-documentation/SDK/bedrock-python-sdk.zip -d /root/bedrock

#!pip install --upgrade pip
#!pip install scikit-learn seaborn

#!pwd
#!python3 -m pip install /root/bedrock/boto3-1.26.142-py3-none-any.whl
#!python3 -m pip install /root/bedrock/botocore-1.29.142-py3-none-any.whl

# In[14]:


!pwd
!python3 -m pip install boto3-1.26.142-py3-none-any.whl --quiet
!python3 -m pip install botocore-1.29.142-py3-none-any.whl --quiet

# In[20]:


import boto3
import json
import csv
from datetime import datetime

bedrock = boto3.client(
 service_name='bedrock',
 region_name='us-east-1',
 endpoint_url='https://bedrock.us-east-1.amazonaws.com'
)

# In[21]:


bedrock.list_foundation_models()

# In[74]:


def call_bedrock(modelId, prompt_data):
    if 'amazon' in modelId:
        body = json.dumps({"inputText": prompt_data})
        #modelId = 'amazon.titan-tg1-large'
    elif 'anthropic' in modelId:
        body = json.dumps({"prompt": prompt_data, "max_tokens_to_sample": 4096})
        #modelId = 'anthropic.claude-instant-v1'
    elif 'ai21' in modelId:
        body = json.dumps({"prompt": prompt_data,
            "maxTokens":4096,
            "stopSequences":[],
            "temperature":0,
            "topP":0.9
        })
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
    with open('prompt-data.csv', 'a') as f:
        writer = csv.writer(f)
        #writer.writerow(column_name)
        writer.writerow(data)
    
    return response, latency

# ### Titan

# In[29]:


prompt_data = """
Generate a list of reasons why customers should have a Medicine Ball at home using the following data
Weighted medicine ball for upper, lower, and full body exercises
The ball has a lightly textured surface provides a superior grip
Weight clearly labeled on both sides.
Ideal for classic medicine ball workouts, including ball toss, weighted twists, squats, sit ups, and more
The person tool to help you develop core strength, balance, and coordination.

"""

# In[30]:


response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ### Claude

# In[97]:


prompt_data = """Human: Generate a list of reasons why customers should have a Medicine Ball at home using the data inside <metadata></metadata> XML tags.

<metadata>
Weighted medicine ball for upper, lower, and full body exercises
The ball has a lightly textured surface provides a superior grip
Weight clearly labeled on both sides.
Ideal for classic medicine ball workouts, including ball toss, weighted twists, squats, sit ups, and more
The person tool to help you develop core strength, balance, and coordination.
</metadata>

Asistant:
"""

# In[98]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ### Jurassic-2

# In[32]:


prompt_data ="""Generate a list of reasons why customers should have a Medicine Ball at home using the following data: Weighted medicine ball for upper, lower, and full body exercises
The ball has a lightly textured surface provides a superior grip. Weight clearly labeled on both sides. Ideal for classic medicine ball workouts, including ball toss, weighted twists, squats, sit ups, and more. The person tool to help you develop core strength, balance, and coordination."""

# In[31]:


prompt_data = """
Generate a description for a red t-shirt

"""

# In[54]:


response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[32]:


response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[33]:


prompt_data = """
Human: Generate a description for a red t-shirt
Assistant:
"""

# In[34]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[35]:


prompt_data = """
Human: Generate a description for a red t-shirt
Assistant:
"""

# In[36]:


response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[131]:


prompt_data = """
Complete the following instructions in order
Intructions:
1. Generate a description for red t-shirt
2. Createa description for a blue hat
3. Generate one reason why people should wear a blue hat

"""

# In[132]:


response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[139]:


prompt_data = """Human: Follow the instruction inside <metadata></metadata> XML tags.

<metadata>
1. Generate a description for red t-shirt
2. Create description for a blue hat
3. Generate one reason why people should wear a blue hat
</metadata>

Asistant:
"""

# In[140]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[141]:


prompt_data ="""Execute the following instructions: 
1. Generate a description for red t-shirt
2. Create description for a blue hat
3. Generate one reason why people should wear a blue hat
"""

# In[142]:


response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[43]:


prompt_data = """
Summarize the following text:
Weighted medicine ball for upper, lower, and full body exercises. The ball has a lightly textured surface provides a superior grip. The Weight clearly labeled on both sides.
Its ideal for classic medicine ball workouts, including ball toss, weighted twists, squats, sit ups, and more. The personal tool to help you develop core strength, balance, and coordination.



"""

# In[44]:


response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[45]:


prompt_data = """Human: Provide a summary of the text inside the <metadata></metadata> XML tags.

<metadata>
Weighted medicine ball for upper, lower, and full body exercises. The ball has a lightly textured surface provides a superior grip. The Weight clearly labeled on both sides.
Its ideal for classic medicine ball workouts, including ball toss, weighted twists, squats, sit ups, and more. The personal tool to help you develop core strength, balance, and coordination.
</metadata>

Asistant:
"""

# In[46]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[47]:


prompt_data = """
Summarize the following text:
Weighted medicine ball for upper, lower, and full body exercises. The ball has a lightly textured surface provides a superior grip. The Weight clearly labeled on both sides.
Its ideal for classic medicine ball workouts, including ball toss, weighted twists, squats, sit ups, and more. The personal tool to help you develop core strength, balance, and coordination.



"""

# In[48]:


response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[51]:


prompt_data = """
Here is an ad for a product, please extract the product name and what its use for :

Weighted medicine ball for upper, lower, and full body exercises. The ball has a lightly textured surface provides a superior grip. The Weight clearly labeled on both sides.
Its ideal for classic medicine ball workouts, including ball toss, weighted twists, squats, sit ups, and more. The personal tool to help you develop core strength, balance, and coordination.



"""

# In[52]:


response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[53]:


prompt_data = """Human:
The information about a new product is inside the <metadata></metadata> XML tags, I want you to extract the product name and what its use for

<metadata>
Weighted medicine ball for upper, lower, and full body exercises. The ball has a lightly textured surface provides a superior grip. The Weight clearly labeled on both sides.
Its ideal for classic medicine ball workouts, including ball toss, weighted twists, squats, sit ups, and more. The personal tool to help you develop core strength, balance, and coordination.
</metadata>

Asistant:
"""

# In[54]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[57]:


prompt_data = """
Here is an description of product, extract the product name and what its use for :

Weighted medicine ball for upper, lower, and full body exercises. The ball has a lightly textured surface provides a superior grip. The Weight clearly labeled on both sides.
Its ideal for classic medicine ball workouts, including ball toss, weighted twists, squats, sit ups, and more. The personal tool to help you develop core strength, balance, and coordination.



"""

# In[58]:


response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[75]:


prompt_data ="""Generate a list of reasons why customers should have a Medicine Ball at home using the following data in 2 sentences: Weighted medicine ball for upper, lower, and full body exercises
The ball has a lightly textured surface provides a superior grip. Weight clearly labeled on both sides. Ideal for classic medicine ball workouts, including ball toss, weighted twists, squats, sit ups, and more. The person tool to help you develop core strength, balance, and coordination."""

# In[76]:


response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[ ]:




# In[ ]:



