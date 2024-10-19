#!/usr/bin/env python
# coding: utf-8

# In[3]:


%pip install transformers==4.31.0 \
             datasets==2.14.4 \
             diffusers==0.20.0 \
             accelerate==0.21.0 \
             torch==2.0.1 \
             torchvision

# In[4]:


from diffusers import StableDiffusionPipeline
import torch

#model_name = "stabilityai/stable-diffusion-xl-base-1.0"
#model_name = "CompVis/stable-diffusion-v1-4"
model_name = "runwayml/stable-diffusion-v1-5"

sd_pipeline = StableDiffusionPipeline.from_pretrained(model_name,
                                                      #torch_dtype=torch.float16
                                                     )

# In[5]:


from datasets import load_dataset

# prompts = load_dataset("nateraw/parti-prompts", split="train")
# prompts = prompts.shuffle()
# sample_prompts = [prompts[i]["Prompt"] for i in range(5)]

# Fixing these sample prompts in the interest of reproducibility.
sample_prompts = [
    "a corgi",
    "a hot air balloon with a yin-yang symbol, with the moon visible in the daytime sky",
    "a car with no windows",
    "a cube made of porcupine",
    'The saying "BE EXCELLENT TO EACH OTHER" written on a red brick wall with a graffiti image of a green alien wearing a tuxedo. A yellow fire hydrant is on a sidewalk in the foreground.',
]

# In[6]:


import torch

seed = 0
generator = torch.manual_seed(seed)

images = sd_pipeline(sample_prompts, 
                     num_images_per_prompt=1, 
                     generator=generator, 
                     output_type="numpy").images

# In[7]:


images

# In[ ]:



