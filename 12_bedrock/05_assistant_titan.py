#!/usr/bin/env python
# coding: utf-8

# # Question and Answer Assistant with Amazon Bedrock and Titan
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 
# Question Answering (QA) is an important task that involves extracting answers to factual queries posed in natural language. Typically, a QA system processes a query against a knowledge base containing structured or unstructured data and generates a response with accurate information. Ensuring high accuracy is key to developing a useful, reliable and trustworthy question answering system, especially for enterprise use cases. 
# 
# Generative AI models like Titan and Claude use probability distributions to generate responses to questions. These models are trained on vast amounts of text data, which allows them to predict what comes next in a sequence or what word might follow a particular word. However, these models are not able to provide accurate or deterministic answers to every question because there is always some degree of uncertainty in the data. Enterprises need to query domain specific and proprietary data and use the information to answer questions, and more generally data on which the model has not been trained on. 
# 
# In this module, we will demonstrate how to use the Bedrock Titan model to provide information response to queries.
# 
# In this example we will be running the Model with no context and then manually try and provide the context. There is no `RAG` augmentation happening here. This approach works with short documents or single-ton applications, it might not scale to enterprise level question answering where there could be large enterprise documents which cannot all be fit into the prompt sent to the model. 
# 
# ### Challenges
# - How to have the model return factual answers for question
# 
# ### Proposal
# To the above challenges, this notebook proposes the following strategy
# #### Prepare documents
# Before being able to answer the questions, the documents must be processed and a stored in a document store index
# - Here we will send in the request with the full relevant context to the model and expect the response back
# 

# ## Setup

# In[2]:


# %pip install boto3 botocore --force-reinstall --quiet
# %pip install langchain==0.0.309 --quiet 

# In[3]:


#### Un comment the following lines to run from your local environment outside of the AWS account with Bedrock access

#import os
#os.environ['BEDROCK_ASSUME_ROLE'] = '<YOUR_VALUES>'
#os.environ['AWS_PROFILE'] = '<YOUR_VALUES>'

# In[4]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# ## Section 1: Q&A with the knowledge of the model
# In this section we try to use models provided by Bedrock service to answer questions based on the knowledge it gained during the training phase.

# In this Notebook we will be using the `invoke_model()` method of Amazon Bedrock client. The mandatory parameters required to use this method are `modelId` which represents the Amazon Bedrock model ARN, and `body` which is the prompt for our task. The `body` prompt changes depending on the foundation model provider selected. We walk through this in detail below
# 
# ```
# {
#    modelId=model_id,
#    contentType="application/json",
#    accept="application/json",
#    body=body
# }
# 
# ```

# 
# ## Scenario
# 
# We are trying to model a situation where we are asking the model to provide information to change the tires. We will first ask the model based on the training data to provide us with an answer for our specific make and model of the car. This technique is called 'Zero Shot` . We will soon realize that even though the model seems to be returning the answers which seem relevant to our specific car, it is actually halucinating. The reason we can find that out is because we run through a fake car and we get almost similiar scenario and answer back
# 
# This situation implies we need to augment the model's training with additional data about our specific make and model of the car and then the model will return us very specific answer. In this notebook we will not use any external sources to augment the data but simulate how a RAG based augmentation system would work. 
# 
# To run our final test we provide a full detailed section from our manual which goes and explains for our specific car how the tire changes work and then we will test to get a curated response back from the model
# 
# ## Task
# 
# To start the process, you select one of the models provided by Bedrock. For this use case you select Titan. These models are able to answer generic questions about cars.
# 
# For example you ask the Titan model to tell you how to change a flat tire on your Audi.
# 

# In[5]:


prompt_data = """You are an helpful assistant. Answer questions in a concise way. If you are unsure about the
answer say 'I am unsure'

Question: How can I fix a flat tire on my Audi A8?
Answer:"""
parameters = {
    "maxTokenCount":512,
    "stopSequences":[],
    "temperature":0,
    "topP":0.9
    }

# #### Let's invoke of the model passing in the JSON body to generate the response

# In[7]:


body = json.dumps({"inputText": prompt_data, 
                   "textGenerationConfig": parameters})

modelId = "amazon.titan-text-express-v1"
accept = "application/json"
contentType = "application/json"

response = bedrock_runtime.invoke_model(
    body=body, 
    modelId=modelId, 
    accept=accept, 
    contentType=contentType
)
response_body = json.loads(response.get("body").read())
answer = response_body.get("results")[0].get("outputText")
print(answer.strip())

# The model is able to give you an answer outlining the process of changing the car flat tire, but the same explanation could be valid for any car. Unfortunately this is not the right answer for an Audi A8, which does not have a spare tire. This is because the model has been trained on data that contains instructions about changing tires on cars.
# 
# Another example of this issue can be seen by trying to ask the same question for a completely fake car brand and model, say a Amazon Tirana.

# In[ ]:


prompt_data = "How can I fix a flat tire on my Amazon Tirana?"
body = json.dumps({"inputText": prompt_data, "textGenerationConfig": parameters})
modelId = "amazon.titan-text-express-v1"
accept = "application/json"
contentType = "application/json"

response = bedrock_runtime.invoke_model(
    body=body, modelId=modelId, accept=accept, contentType=contentType
)
response_body = json.loads(response.get("body").read())
answer = response_body.get("results")[0].get("outputText")
print(answer.strip())

# As you can see the answer that the model provides is plausible but it is for a bike and not a car. The model assumed that the Amazon Tirana was a bike. The model is _hallucinating_.
# 
# How can we fix this issue and have the model provide answers based on the specific instructions valid for my car model?
# 
# Research by Facebook in 2020 found that LLM knowledge could be augmented on the fly by providing the additional knowledge base as part of the prompt. This approach is called Retrieval Augmented Generation, or RAG.
# 
# Let's see how we can use this to improve our application.
# 
# The following is an excerpt of the manual of the Audi A8 (in reality it is not the real manual, but let's assume so). This document is also conveniently short enough to fit entirely in the prompt of Titan Large. 
# 
# ```
# Tires and tire pressure:
# 
# Tires are made of black rubber and are mounted on the wheels of your vehicle. They provide the necessary grip for driving, cornering, and braking. Two important factors to consider are tire pressure and tire wear, as they can affect the performance and handling of your car.
# 
# Where to find recommended tire pressure:
# 
# You can find the recommended tire pressure specifications on the inflation label located on the driver's side B-pillar of your vehicle. Alternatively, you can refer to your vehicle's manual for this information. The recommended tire pressure may vary depending on the speed and the number of occupants or maximum load in the vehicle.
# 
# Reinflating the tires:
# 
# When checking tire pressure, it is important to do so when the tires are cold. This means allowing the vehicle to sit for at least three hours to ensure the tires are at the same temperature as the ambient temperature.
# 
# To reinflate the tires:
# 
#     Check the recommended tire pressure for your vehicle.
#     Follow the instructions provided on the air pump and inflate the tire(s) to the correct pressure.
#     In the center display of your vehicle, open the "Car status" app.
#     Navigate to the "Tire pressure" tab.
#     Press the "Calibrate pressure" option and confirm the action.
#     Drive the car for a few minutes at a speed above 30 km/h to calibrate the tire pressure.
# 
# Note: In some cases, it may be necessary to drive for more than 15 minutes to clear any warning symbols or messages related to tire pressure. If the warnings persist, allow the tires to cool down and repeat the above steps.
# 
# Flat Tire:
# 
# If you encounter a flat tire while driving, you can temporarily seal the puncture and reinflate the tire using a tire mobility kit. This kit is typically stored under the lining of the luggage area in your vehicle.
# 
# Instructions for using the tire mobility kit:
# 
#     Open the tailgate or trunk of your vehicle.
#     Lift up the lining of the luggage area to access the tire mobility kit.
#     Follow the instructions provided with the tire mobility kit to seal the puncture in the tire.
#     After using the kit, make sure to securely put it back in its original location.
#     Contact Rivesla or an appropriate service for assistance with disposing of and replacing the used sealant bottle.
# 
# Please note that the tire mobility kit is a temporary solution and is designed to allow you to drive for a maximum of 10 minutes or 8 km (whichever comes first) at a maximum speed of 80 km/h. It is advisable to replace the punctured tire or have it repaired by a professional as soon as possible.
# ```
# 
#  
# Next, we take this text and "embed" it in the prompt together with the original question. The prompt is also build in such a way as to try to hint the model to only look at the information provided as context.

# In[9]:


context = """Tires and tire pressure:

Tires are made of black rubber and are mounted on the wheels of your vehicle. They provide the necessary grip for driving, cornering, and braking. Two important factors to consider are tire pressure and tire wear, as they can affect the performance and handling of your car.

Where to find recommended tire pressure:

You can find the recommended tire pressure specifications on the inflation label located on the driver's side B-pillar of your vehicle. Alternatively, you can refer to your vehicle's manual for this information. The recommended tire pressure may vary depending on the speed and the number of occupants or maximum load in the vehicle.

Reinflating the tires:

When checking tire pressure, it is important to do so when the tires are cold. This means allowing the vehicle to sit for at least three hours to ensure the tires are at the same temperature as the ambient temperature.

To reinflate the tires:

    Check the recommended tire pressure for your vehicle.
    Follow the instructions provided on the air pump and inflate the tire(s) to the correct pressure.
    In the center display of your vehicle, open the "Car status" app.
    Navigate to the "Tire pressure" tab.
    Press the "Calibrate pressure" option and confirm the action.
    Drive the car for a few minutes at a speed above 30 km/h to calibrate the tire pressure.

Note: In some cases, it may be necessary to drive for more than 15 minutes to clear any warning symbols or messages related to tire pressure. If the warnings persist, allow the tires to cool down and repeat the above steps.

Flat Tire:

If you encounter a flat tire while driving, you can temporarily seal the puncture and reinflate the tire using a tire mobility kit. This kit is typically stored under the lining of the luggage area in your vehicle.

Instructions for using the tire mobility kit:

    Open the tailgate or trunk of your vehicle.
    Lift up the lining of the luggage area to access the tire mobility kit.
    Follow the instructions provided with the tire mobility kit to seal the puncture in the tire.
    After using the kit, make sure to securely put it back in its original location.
    Contact Audi or an appropriate service for assistance with disposing of and replacing the used sealant bottle.

Please note that the tire mobility kit is a temporary solution and is designed to allow you to drive for a maximum of 10 minutes or 8 km (whichever comes first) at a maximum speed of 80 km/h. It is advisable to replace the punctured tire or have it repaired by a professional as soon as possible."""

# #### Let's take the whole excerpt and pass it to the model together with the question.

# In[10]:


question = "How can I fix a flat tire on my Audi A8?"
prompt_data = f"""Answer the question based only on the information provided between ## and give step by step guide.
#
{context}
#

Question: {question}
Answer:"""

# ##### Invoke the model via boto3 to generate the response

# In[11]:


body = json.dumps({"inputText": prompt_data, "textGenerationConfig": parameters})
modelId = "amazon.titan-text-express-v1"
accept = "application/json"
contentType = "application/json"

response = bedrock_runtime.invoke_model(
    body=body, modelId=modelId, accept=accept, contentType=contentType
)
response_body = json.loads(response.get("body").read())
answer = response_body.get("results")[0].get("outputText")
print(answer.strip())

# Since the model takes a while to understand the context and generate relevant answer for you, this might lead to poor experience for the user since they have to wait for a response for some seconds.
# 
# Bedrock also supports streaming capability where the service generates an output as the model is generating tokens. Here is an example of how you can do that.

# In[12]:


from IPython.display import display_markdown,Markdown,clear_output

# In[13]:


response = bedrock_runtime.invoke_model_with_response_stream(body=body, modelId=modelId, accept=accept, contentType=contentType)
stream = response.get('body')
output = []
i = 1
if stream:
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['outputText']
            clear_output(wait=True)
            output.append(text)
            display_markdown(Markdown(''.join(output)))
            i+=1

# ## Summary
# 
# We see the response is a summarized and step by step instruction of how to change the tires . This simple example shows how you can leverage the `RAG` or the Augmentation process to generate a curated response back

# In[ ]:



