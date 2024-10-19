#!/usr/bin/env python
# coding: utf-8

# ## Textual Inversion fine-tuning Stable Diffusion
# 
# [Textual inversion](https://arxiv.org/abs/2208.01618) is a method to personalize text2image models like stable diffusion on your own images using just 3-5 examples.
# The `textual_inversion.py` script shows how to implement the training procedure and adapt it for stable diffusion.
# 
# Here, we'll customize Stable Diffusion by fine-tuning with Molly pictures using an example adapted from Hugging Face.  To do this, we'll utilize some Hugging Face libraries including the [Diffuser's Library](https://huggingface.co/docs/diffusers/index) and [PEFT Libraries](https://huggingface.co/docs/peft/index).  
# 
# To start, we'll install some prerequisites and perform some initial setup. 

# ## Setup

# In[3]:


!pip install torch==2.0.1 \
  accelerate==0.24.1 \
  transformers==4.35.2 \
  torchvision \
  ftfy \
  tensorboard \
  Jinja2 \
  ipywidgets

# In[11]:


!pip install git+https://github.com/huggingface/diffusers

# In[12]:


!pip uninstall xformers -y

# In[13]:


!conda install -y xformers -c xformers

# ## Perform Inference
# 
# Next, we can perform inference using our learned embeddings combined with a prompt that contains the learned token 'M*'

# In[14]:


from diffusers import StableDiffusionPipeline
import torch

# In[16]:


pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# In[17]:


pipe.load_textual_inversion("./textual_inversion_molly/learned_embeds.bin", token="M*")

# In[18]:


prompt = "An oil painting of M*"

image = pipe(prompt, num_inference_steps=50).images[0]
image.save("molly-dog3.png")

# In[19]:


from IPython.display import Image
Image("./molly-dog3.png")

# In[14]:


from diffusers import StableDiffusionPipeline
import torch

# In[16]:


pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# In[17]:


pipe.load_textual_inversion("./textual_inversion_molly/learned_embeds.bin", token="M*")

# In[18]:


prompt = "An oil painting of M*"

image = pipe(prompt, num_inference_steps=50).images[0]
image.save("molly-dog3.png")

# In[19]:


from IPython.display import Image
Image("./molly-dog3.png")

# In[ ]:



