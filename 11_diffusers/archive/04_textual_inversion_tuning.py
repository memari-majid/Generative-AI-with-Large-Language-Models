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

# In[2]:


!pip install torch==2.0.1 \
  accelerate==0.24.1 \
  transformers==4.35.2 \
  torchvision \
  ftfy \
  tensorboard \
  Jinja2 \
  ipywidgets

# In[ ]:


!pip install git+https://github.com/huggingface/diffusers

# In[6]:


!pip uninstall xformers -y

# In[7]:


!conda install -y xformers -c xformers

# In[5]:


from accelerate.utils import write_basic_config

write_basic_config()

# Training Script Parameters:
# 
# It is often a good idea to regularly save checkpoints of your model during training. This way, you can resume training from a saved checkpoint if your training is interrupted for any reason. To save a checkpoint, pass the following argument to the training script to save the full training state in a subfolder in output_dir every 500 steps:
# 
# --checkpointing_steps=500
# 
# 
# To resume training from a saved checkpoint, pass the following argument to the training script and the specific checkpoint you’d like to resume from:
# 
# --resume_from_checkpoint="checkpoint-1500"
# 
# 
# This guide will show you two ways to create a dataset to finetune on:
# 
# provide a folder of images to the --train_data_dir argument
# upload a dataset to the Hub and pass the dataset repository id to the --dataset_name argument
# 

# In[11]:


local_dir = "./images-ti-molly"

# In[17]:


# learnable_property - valid options are 'object' or style

!accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="./images-ti-molly" \
  --learnable_property="object" \
  --placeholder_token="M*" \
  --initializer_token="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_molly"

# In[ ]:



