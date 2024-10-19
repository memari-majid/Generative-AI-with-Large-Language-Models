#!/usr/bin/env python
# coding: utf-8

# # Create Images with Amazon Titan Image Generator

# In[2]:


import boto3
import json
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime

# In[3]:


bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# ## List Bedrock Models by Provider

# In[4]:


response = bedrock.list_foundation_models(
    byProvider="amazon",
    byOutputModality="IMAGE",
)

# In[5]:


for model in response["modelSummaries"]:
    print("-----\n" + "modelArn: " + model["modelArn"] + "\nmodelId: " + model["modelId"] + "\nmodelName: " + model["modelName"] + "\ncustomizationsSupported: " + ','.join(model["customizationsSupported"]))

# # Create Images

# In[6]:


model_id = 'amazon.titan-image-generator-v1'

# ## Example: Text to Image
# ImageGenerationConfig Options:
# * numberOfImages: Number of images to be generated
# * quality: Quality of generated images, can be standard or premium
# * height: Height of output image(s)
# * width: Width of output image(s)
# * cfgScale: Scale for classifier-free guidance
# * seed: The seed to use for re-producibility  

# In[7]:


body = json.dumps(
    {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": "Generate a green iguana on a tree branch.",   # Required
            # "negativeText": "<text>"  # Optional
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,      # Range: 1 to 5 
            "quality": "premium",     # Options: standard/premium
            "height": 1024,
            "width": 1024,
            "cfgScale": 7.5,          # Range: 1.0 (exclusive) to 10.0
            "seed": 42                # Range: 0 to 214783647
        }
    }
)

# In[8]:


accept = "application/json"
contentType = "application/json"

# In[9]:


%%time

response = bedrock_runtime.invoke_model(
    body=body, modelId=model_id, accept=accept, contentType=contentType
)
response_body = json.loads(response.get("body").read())

# In[10]:


images = [Image.open(BytesIO(base64.b64decode(base64_image))) for base64_image in response_body.get("images")]
for img in images:
    img.save("iguana.png")
    display(img)

# In[11]:


image_path = "iguana.png"

# In[12]:


with open(image_path, "rb") as image_file:
    input_image = base64.b64encode(image_file.read()).decode('utf8')

# In[13]:


caption = """
Change the animal color to orange.
"""

body = json.dumps(
    {
        "taskType": "INPAINTING",
        "inPaintingParams": {
            "text": "Change the lizard color to orange.", 
            # "negativeText": "bad quality, low res",
            "image": input_image,
            "maskPrompt": "animal",
            # "maskImage": None, # One of "maskImage" or "maskPrompt" is required             
        },                                                 
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium",
            "height": 512,
            "width": 512,
            "cfgScale": 8.0
        }
    }
)

# In[14]:


%%time

response = bedrock_runtime.invoke_model(
    body=body, modelId=model_id, accept=accept, contentType=contentType
)
response_body = json.loads(response.get("body").read())

images = [Image.open(BytesIO(base64.b64decode(base64_image))) for base64_image in response_body.get("images")]

print(f"Generated {len(images)} images...")
for img in images:
    img.save("inpainting_iguana.png")
    display(img)

# In[15]:


# Out-painting
# 
# OutPaintingParams Options:
#   text: prompt to guide out-painting
#   negativeText: prompts to guide the model on what you dont want in image 
#   image: base64 string representation of the input image
#   maskImage: base64 string representation of the input mask image
#   maskPrompt: prompt used for auto editing to generate mask
#   outPaintingMode: DEFAULT || PRECISE   
body = json.dumps(
    {
        "taskType": "OUTPAINTING",
        "outPaintingParams": {
            "text": "Change the background",          # Required
            #"negativeText": "bad quality, low res",  # Optional
            "image": input_image,                     # Required
            "maskPrompt": "background",               # One of "maskImage" or "maskPrompt" is required
            #"maskImage": None,                       # One of "maskImage" or "maskPrompt" is required 
            "outPaintingMode": "DEFAULT"              # Optional  
        },                                                 
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium",
            "height": 512,
            "width": 512,
            "cfgScale": 8.0
        }
    }
)

# In[16]:


%%time

response = bedrock_runtime.invoke_model(
    body=body, modelId=model_id, accept=accept, contentType=contentType
)
response_body = json.loads(response.get("body").read())

images = [Image.open(BytesIO(base64.b64decode(base64_image))) for base64_image in response_body.get("images")]

print(f"Generated {len(images)} images...")
for img in images:
    img.save("outpainting_iguana.png")
    display(img)

# In[17]:


# Image Variation 
# ImageVariationParams Options:
#  text: prompt to guide the model on how to generate variations
#  negativeText: prompts to guide the model on what you don't want in image
#  images: base64 string representation of the input image, only 1 is supported  
body = json.dumps(
    {
        "taskType": "IMAGE_VARIATION",
        "imageVariationParams": {
            "text": "Convert the animal to a photo-realistic, 8k, hdr",  # Required
            "negativeText": "bad quality, low resolution, cartoon",   # Optional
            "images": [input_image],                                  # One image is required
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium",
            "height": 512,
            "width": 512,
            "cfgScale": 8.0
        }
    }
)

# In[18]:


%%time

response = bedrock_runtime.invoke_model(
    body=body, modelId=model_id, accept=accept, contentType=contentType
)
response_body = json.loads(response.get("body").read())

images = [Image.open(BytesIO(base64.b64decode(base64_image))) for base64_image in response_body.get("images")]

print(f"Generated {len(images)} images...")
for img in images:
    img.save("variation_iguana.png")
    display(img)

# In[ ]:



