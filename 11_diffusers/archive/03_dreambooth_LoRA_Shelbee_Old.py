#!/usr/bin/env python
# coding: utf-8

# # Fine-tune Stable Diffusion using Dreambooth with LoRA

# In this notebook, we show how to fine-tune a Stable Diffusion model using your own dataset but specifically utilizing the low rank adaptation technique, LoRA, which allows us to fine-tune the cross-attention layers without performing full fine-tuning. 
# 
# To do this, we'll utilize some Hugging Face libraries including the [Diffuser's Library](https://huggingface.co/docs/diffusers/index) and [PEFT Libraries](https://huggingface.co/docs/peft/index). 
# 
# Once again, we'll focus on fine-tuning using a few images of Molly dog as a puppy! 

# # Setup
# 
# First, we'll perform some setup by installing dependencies.  Please restart the kernel after all dependencies have been installed.

# In[2]:


!pip install transformers accelerate>=0.20.3 ftfy tensorboard Jinja2 huggingface_hub wandb kaggle git+https://github.com/huggingface/diffusers

# In[3]:


! pip install -U accelerate
! pip install -U transformers

# In[4]:


import accelerate
import transformers

transformers.__version__, accelerate.__version__

# In[5]:


!wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora.py

# In[6]:


!wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/requirements.txt

# In[7]:


!pip install -r requirements.txt
!pip install git+https://github.com/huggingface/peft

# In[8]:


!pwd

# In[9]:


#!export INSTANCE_DIR='/home/sagemaker-user/dreambooth_lora/training_images'
#!export CLASS_DIR='/home/sagemaker-user/dreambooth_lora/base_class'

# In[10]:


import shutil, os
dog_images = os.listdir("images-lora-molly")

# In[11]:


!mkdir -p images-lora-molly-p
!mkdir -p images-lora-base-class

# In[12]:


for filename in dog_images:
    shutil.copyfile(
         os.path.join("images-lora-molly-p",filename),
         os.path.join("images-lora-base-class",filename)
   )

# In[ ]:


%autosave 0

# In[13]:


instance_data_directory = '/home/sagemaker-user/generative-ai-on-aws/11_controlnet/images-lora-molly-p'

# Next, we'll use a training script adapted from a Hugging Face example to fine-tune a Stable Diffusion model, specifically verson 1.5, using LoRA fine-tuning.  Also, to show variety in how you can choose to uilitize SageMaker as well as common libraries, we are showing training being locally performed on this SageMaker Studio notebook instance versus separate compute through spinning up training jobs.  This can be a good option for experimentation and you can easily [change the size of your SageMaker notebook](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-run-and-manage-switch-instance-type.html) on the fly to support this type of experimentation.
# 
# We'll be using another Hugging Face library called [Accelerate](https://huggingface.co/docs/accelerate/index) which provides a variety of helpful libraries for running training and inference.
# 
# Note, in this case we are specifying the instance and class prompt directly in the configuration supplied instead of separately in the dataset_info.json as previously shown in the full fine-tuning notebook (03_dreambooth.ipynb).   
# 
# A few parameters to point out below include: 
# 
#   * **lora_r**: This parameter indicates the rank of the low rank matrices where a lower rank results in smaller matrices with fewer trainable parameters.
#   * **lora_alpha**: This parameter notes the scaling factor to be used and controls how much importance you want to give to the new updated weight matrices when combining it with the original pretrained weights. 
#   * **lr_scheduler**: This parameter controls how the learning rate changes during training. In the example below, we set it to 'constant' meaning that the learning rate will remain the same during training. 
#   * **lr_warmup_steps**: This parameter specifies the number of warmup steps.

# In[14]:


#https://huggingface.co/docs/peft/task_guides/dreambooth_lora#finetuning-dreambooth
!accelerate launch train_dreambooth_hf_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
  --instance_data_dir="images-lora-molly-p" \
  --class_data_dir="images-lora-base-class" \
  --output_dir=trained_model \
  --train_text_encoder \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of molly dog" \
  --class_prompt="a photo of a sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 27 \
  --lora_text_encoder_r 16 \
  --lora_text_encoder_alpha 17 \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=800

# ## Run inference against fine-tuned weights
# 
# Next, let's run inference against the fine-tune weights using the base model and the new fine-tuned LoRA weights.

# In[ ]:


from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

# In[ ]:


pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# In[ ]:


def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="cuda", adapter_name="molly"
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe

# In[ ]:


from pathlib import Path
from peft import PeftModel, LoraConfig
from diffusers import StableDiffusionPipeline

pipe = get_lora_sd_pipeline(Path("./trained_model"), adapter_name="molly")

# In[ ]:


from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# Now, let's prompt the model (in this case the base foundation model combined with the LoRA adapter weights) for a picture of Molly on the beach. 

# In[ ]:


img_list = pipe(["Molly dog on a beach"]*3, num_inference_steps=50).images
image_grid([x.resize((128,128)) for x in img_list], 1,3)

# # Congratulations! 
# 
# You fine-tuned Stable Diffusion models using a parameter efficient, light weight method that still allows for customization of the model based on your own dataset.  It's also suggested to review the size of the fine-tuned weights.  You'll notice the the fine-tuned weights are a fraction of the size of the full model allowing you to easily customize the model across a number of custom classes or subjects.  
# 
# In this case, we used our photos of Molly dog to fine-tune but feel free to try out your own examples.

# In[ ]:



