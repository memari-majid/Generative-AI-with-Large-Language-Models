#!/usr/bin/env python
# coding: utf-8

# # Introduction to Bedrock - PE - Auto

# Before moving on with this notebook, you should finish install dependencies

# ### Environment Prep

# In[3]:


import boto3
import json
import csv
from datetime import datetime

bedrock = boto3.client(
 service_name='bedrock',
 region_name='us-east-1',
 endpoint_url='https://bedrock.us-east-1.amazonaws.com'
)

# In[4]:


bedrock.list_foundation_models()

# ### Util Functions

# In[19]:


## This function will help you invoke the model and then save the following meta data in to a csv:
## Timestamp, Model Used, Prompt, Result, Latency
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
        body = json.dumps({"prompt": prompt_data, 
                           "max_tokens_to_sample": 8000})
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
    data = [datetime.now(), modelId, prompt_data, response, latency] #the data
    with open('./prompt-data/prompt-data.csv', 'a') as f:
        writer = csv.writer(f)
        #writer.writerow(column_name)
        writer.writerow(data)
    
    return response, latency

# In[6]:


!mkdir prompt-data

# # Zero-Shot QA 

# #### Use Case : 
# Suppose we want LLM to give some recommendations about vehicles, and we have 2 requirements:
# 
# 1, price range: 20000-40000
# 
# 2, number of seats: 5

# ### Titan

# In[12]:


prompt_data ="""
Generate some recommendations for vehicles that satisfy the following requirements:
price range: 20000-40000
number of seats: 5
"""
response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print("Prompt: ", "\n", prompt_data)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# #### By default, Titan tends to like short, precise, well formatted prompts and give short answers, but we can instruct it to give more details.

# In[15]:


prompt_data ="""
Generate 2 recommendations for vehicles that satisfy the following requirements, and explain:
price range: 20000-40000
number of seats: 5
"""

response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print("Prompt: ", "\n", prompt_data)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# ### Claude

# In[18]:


prompt_data ="""Human:
Generate some recommendations for vehicles that satisfy the requirements in the <metadata></metadata> XML tags.
<metadata>
price range: 20000-40000
number of seats: 5
</metadata>
Assistant:
"""

response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print("Prompt: ", "\n", prompt_data)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# #### By default, Claude tends to be chatty, and it goes both ways, it also takes chatty prompt very well.

# In[28]:


prompt_data ="""Human:
Provide 2 recommendations for vehicles that satisfy the following requirements, and provide detailed information:
price range: 20000-40000
number of seats: 5
Assistant:
"""

response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print("Prompt: ", "\n", prompt_data)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# ### Jurassic-2

# In[30]:


prompt_data ="""
Generate some recommendations for vehicles that satisfy the requirements based on the following requirements: 
price range: 20000-50000, 
number of seats: 5.
"""

print("Prompt: ", "\n", prompt_data)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# #### Jurrasic-2 and Titan have very similar prompt formats, it is relatively robust with different wording.

# In[31]:


prompt_data ="""
Generate 2 recommendations for vehicles that satisfy the requirements based on the following requirements, and explain in details: 
price range: 20000-50000, 
number of seats: 5.
"""

print("Prompt: ", "\n", prompt_data)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# # Zero-Shot QA with Context

# #### Use Case : 
# Suppose we want to ask question about the user guide, which is quite a common use case in the Auto industry.  End users may be confused about a certain feature, but they do not want to go through the entire document or waiting on manual help. 
# 
# ##### For illustration purpose we are going to use the following sample user guide:
# 
# 2023.06.06 Release Notes
# 
# Stopping Mode
# Stopping Mode is now set to "auto" by default after performing a factory reset.
# When Stopping Mode is set to "auto", your car continues to move slowly forward (in Drive) or backwards (in Reverse) when close to a complete stop.
# To change the Stopping Mode, go to Controls > Pedals & Steering > Stopping Mode.
# 
# Refresh Your Data Sharing Settings
# Enable or disable data sharing and learn more about how SomeCar uses anonymous data to improve existing safety and convenience features and create new ones.
# Go to Controls > Software > Data Sharing. For more information on how we protect your data, go to www.sample-data-privacy.com

# In[47]:


context = """
2023.06.06 Release Notes

Stopping Mode
Stopping Mode is now set to "auto" by default after performing a factory reset.
When Stopping Mode is set to "auto", your car continues to move slowly forward (in Drive) or backwards (in Reverse) when close to a complete stop.
To change the Stopping Mode, go to Controls > Pedals & Steering > Stopping Mode.

Refresh Your Data Sharing Settings
Enable or disable data sharing and learn more about how SomeCar uses anonymous data to improve existing safety and convenience features and create new ones.
Go to Controls > Software > Data Sharing. For more information on how we protect your data, go to www.sample-data-privacy.com

"""

# ### Titan

# In[78]:


prompt = """
question1: what is the release note version? question2: how to refresh your data sharing?

Answer previous questions based on the following context:

2023.06.06 Release Notes

Stopping Mode
Stopping Mode is now set to "auto" by default after performing a factory reset.
When Stopping Mode is set to "auto", your car continues to move slowly forward (in Drive) or backwards (in Reverse) when close to a complete stop.
To change the Stopping Mode, go to Controls > Pedals & Steering > Stopping Mode.

Refresh Your Data Sharing Settings
Enable or disable data sharing and learn more about how SomeCar uses anonymous data to improve existing safety and convenience features and create new ones.
Go to Controls > Software > Data Sharing. For more information on how we protect your data, go to www.sample-data-privacy.com

"""
response, latency = call_bedrock('amazon.titan-tg1-large', prompt)
print("Prompt: ", "\n", prompt)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# ### Claude

# In[79]:


prompt = """
Human: I'm going to give you a summary and then I'm going to ask you some questions about the summary.

2023.06.06 Release Notes

Stopping Mode
Stopping Mode is now set to "auto" by default after performing a factory reset.
When Stopping Mode is set to "auto", your car continues to move slowly forward (in Drive) or backwards (in Reverse) when close to a complete stop.
To change the Stopping Mode, go to Controls > Pedals & Steering > Stopping Mode.

Refresh Your Data Sharing Settings
Enable or disable data sharing and learn more about how SomeCar uses anonymous data to improve existing safety and convenience features and create new ones.
Go to Controls > Software > Data Sharing. For more information on how we protect your data, go to www.sample-data-privacy.com

Assitant: 
Here is the first question: what is the release note version? 
Here is the second question: how to refresh your data sharing?? 
"""

response, latency = call_bedrock('anthropic.claude-v1', prompt)
print("Prompt: ", "\n", prompt)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# ### Jurrasic-2

# In[93]:


prompt = """
question1: what is the release note version? question2: how to refresh your data sharing?

Answer previous questions based on the following context:

2023.06.06 Release Notes

Stopping Mode
Stopping Mode is now set to "auto" by default after performing a factory reset.
When Stopping Mode is set to "auto", your car continues to move slowly forward (in Drive) or backwards (in Reverse) when close to a complete stop.
To change the Stopping Mode, go to Controls > Pedals & Steering > Stopping Mode.

Refresh Your Data Sharing Settings
Enable or disable data sharing and learn more about how SomeCar uses anonymous data to improve existing safety and convenience features and create new ones.
Go to Controls > Software > Data Sharing. For more information on how we protect your data, go to www.sample-data-privacy.com
"""
response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt)
print("Prompt: ", "\n", prompt)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# # Zero-Shot Text Generation Based on Context

# #### Use Case : 
# We can utilize LLM to generate Q&A dataset.  This is extremely useful if we want to build a knowledge-based chatbot and fine-tune domain-based LLM.
# 
# ##### For illustration purpose we are going to use the sample user guide above:

# ### Titan

# In[89]:


prompt = """
Generate 3 questions and give answers based on the following information:

2023.06.06 Release Notes

Stopping Mode
Stopping Mode is now set to "auto" by default after performing a factory reset.
When Stopping Mode is set to "auto", your car continues to move slowly forward (in Drive) or backwards (in Reverse) when close to a complete stop.
To change the Stopping Mode, go to Controls > Pedals & Steering > Stopping Mode.

Refresh Your Data Sharing Settings
Enable or disable data sharing and learn more about how SomeCar uses anonymous data to improve existing safety and convenience features and create new ones.
Go to Controls > Software > Data Sharing. For more information on how we protect your data, go to www.sample-data-privacy.com

"""
response, latency = call_bedrock('amazon.titan-tg1-large', prompt)
print("Prompt: ", "\n", prompt)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# ### Claude

# In[92]:


prompt = """
provide 3 questions and answers based on the following information:

2023.06.06 Release Notes

Stopping Mode
Stopping Mode is now set to "auto" by default after performing a factory reset.
When Stopping Mode is set to "auto", your car continues to move slowly forward (in Drive) or backwards (in Reverse) when close to a complete stop.
To change the Stopping Mode, go to Controls > Pedals & Steering > Stopping Mode.

Refresh Your Data Sharing Settings
Enable or disable data sharing and learn more about how SomeCar uses anonymous data to improve existing safety and convenience features and create new ones.
Go to Controls > Software > Data Sharing. For more information on how we protect your data, go to www.sample-data-privacy.com

"""
response, latency = call_bedrock('anthropic.claude-v1', prompt)
print("Prompt: ", "\n", prompt)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# ### Jurrasic-2

# In[96]:


prompt = """
provide 3 questions and answers based on the following information:

2023.06.06 Release Notes

Stopping Mode
Stopping Mode is now set to "auto" by default after performing a factory reset.
When Stopping Mode is set to "auto", your car continues to move slowly forward (in Drive) or backwards (in Reverse) when close to a complete stop.
To change the Stopping Mode, go to Controls > Pedals & Steering > Stopping Mode.

Refresh Your Data Sharing Settings
Enable or disable data sharing and learn more about how SomeCar uses anonymous data to improve existing safety and convenience features and create new ones.
Go to Controls > Software > Data Sharing. For more information on how we protect your data, go to www.sample-data-privacy.com

"""
response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt)
print("Prompt: ", "\n", prompt)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# # Zero-shot Chain of Thought

# #### Use Case : 
# We can utilize LLM to calculate math problems, which may help compare payment plans, discounts and cost in general.

# ### Titan

# #### Note Titan can take both (think step by step) or (show your work) as triggers

# In[107]:


context = """
the total cost of vehicle A is $40000 needs 30% as down payment.
the total cost of vehicle B is $50000 needs 20% as down payment.
(show your work)
"""

#context = """
#the total cost of vehicle A is $40000 needs 30% as down payment.
#the total cost of vehicle B is $50000 needs 20% as down payment.
#(Think Step-by-Step)
#"""

prompt = f"which vehicle needs more down payment? based on the following information:\n\n{context}"
response, latency = call_bedrock('amazon.titan-tg1-large', prompt)
print(prompt)
print(response)

# ### Claude

# #### Note that sometimes Claude can be chatty

# In[105]:


prompt_data ="""
Human: which vehicle needs more down payment? clear but no chatty.  based on the following information:

the total cost of vehicle A is $40000 needs 30% as down payment.
the total cost of vehicle B is $50000 needs 20% as down payment.
(Think Step-by-Step)

Assitant: 
"""

#prompt_data ="""
#Human: which vehicle needs more down payment? based on the following information:

#the total cost of vehicle A is $40000 needs 30% as down payment.
#the total cost of vehicle B is $50000 needs 20% as down payment.
#(Think Step-by-Step)

#Assitant: 
#"""

response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print("Prompt: ", "\n", prompt)
print("\n\n")
print("Response: ", "\n", response)
print("\n\n")
print("Inference time:", "\n", latency)

# ### Jurrasic-2

# #### note that Jurrasic-2 tends to ignore '()'

# In[100]:


context = """
the total cost of vehicle A is $40000 needs 30% as down payment.
the total cost of vehicle B is $50000 needs 20% as down payment.
Think Step-by-Step
"""

#context = """
#the total cost of vehicle A is $40000 needs 30% as down payment.
#the total cost of vehicle B is $50000 needs 20% as down payment.
#(Think Step-by-Step)
"""
prompt = f"which vehicle needs more down payment? based on the following information:\n\n{context}"
response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt)
print(prompt)
print(response)
