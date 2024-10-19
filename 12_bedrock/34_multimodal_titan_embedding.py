#!/usr/bin/env python
# coding: utf-8

# # Create Embeddings with Amazon Titan Multimodal Embeddings

# In[1]:


import boto3
import json
import base64

# In[ ]:


bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# ## List Bedrock Models by Provider

# In[9]:


response = bedrock.list_foundation_models(
    byProvider="amazon",
    byOutputModality="EMBEDDING",
)

# In[10]:


for model in response["modelSummaries"]:
    print("-----\n" + "modelArn: " + model["modelArn"] + "\nmodelId: " + model["modelId"] + "\nmodelName: " + model["modelName"] + "\ncustomizationsSupported: " + ','.join(model["customizationsSupported"]))

# In[11]:


model_id = 'amazon.titan-embed-image-v1'

# ## Create Embeddings

# In[13]:


image_path = "images/iguana.png"

# In[15]:


# MAX image size supported is 2048 * 2048 pixels
with open(image_path, "rb") as image_file:
    input_image = base64.b64encode(image_file.read()).decode('utf8')

caption = """
Iguanas are some of the largest lizards found in the Americas, 
with their whiplike tail making up about half of that length. 
Like other reptiles, iguanas are cold-blooded, egg-laying 
animals with an excellent ability to adapt to their environment.
"""
    
body = json.dumps(
    {
        "inputText": caption,
        "inputImage": input_image
    }
)

# In[16]:


accept = "application/json"
contentType = "application/json"

# In[17]:


response = bedrock_runtime.invoke_model(
    body=body, modelId=model_id, accept=accept, contentType=contentType
)
response_body = json.loads(response.get("body").read())

print(response_body.get("embedding"))

# In[ ]:



