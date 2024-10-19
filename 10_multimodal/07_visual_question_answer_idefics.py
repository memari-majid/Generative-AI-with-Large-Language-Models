#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %pip install -U transformers \
#              datasets==2.14.4 \
#              diffusers==0.20.0 \
#              accelerate==0.21.0 \
#              torch==2.0.1 \
#              torchvision==0.15.2 \
#              sentencepiece==0.1.99

# In[3]:


import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(model_name, 
                                                  torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(model_name)

# Generation args
exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

# # Zero-shot inference

# ![](https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg)

# In[4]:


url = "https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg"
img = processor.image_processor.fetch_images([url])[0]

prompts = [
    "\nUser:",
    img,
    "Describe this image.\nAssistant: ",
]

inputs = processor(prompts, return_tensors="pt", debug=True).to(device)

generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
generated_text

# # One-shot Inference (1 example) to guide the description

# ![](https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg)
# ![](https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg)

# In[5]:


url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
img = processor.image_processor.fetch_images([url])[0]

# Either use img or url
prompts = [
    "User:",
    img,
    "Describe this image."
    "Assistant: An image of two kittens in grass." # One-shot example
    "User:",
    "https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg", 
    "Describe this image.",
    "Assistant: "
]

inputs = processor(prompts, return_tensors="pt", debug=True).to(device)

generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
generated_text

# # Show special characters injected around the images

# In[6]:


generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
generated_text

# # Ask Questions About Text in the Image

# ![](img/happy-car-chris.png)

# In[7]:


from PIL import Image
img = Image.open("img/happy-car-chris.png") 

prompts = [
    "User: ",
    img,
    "Describe this image.",
    "Assistant: ",
]

inputs = processor(prompts, return_tensors="pt", debug=True).to(device)

generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
generated_text

# In[8]:


from PIL import Image
img = Image.open("img/happy-car-chris.png") 

prompts = [
    "User: ",
    img,
    "Who makes this car?",
    "Assistant: ",
]

inputs = processor(prompts, return_tensors="pt", debug=True).to(device)

generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
generated_text

# ![](img/baby-groot-toy.jpg)

# In[9]:


img = Image.open("img/baby-groot-toy.jpg") 

prompts = [
    "User: ",
    img,
    "Which movie is this character from?",
    "Assistant: ",
]

inputs = processor(prompts, return_tensors="pt").to(device)

generated_ids = model.generate(**inputs, max_length=100) # eos_token_id=exit_condition, bad_words_ids=bad_words_ids)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

# # Chain of thought

# ![](img/baby-groot-toy.jpg)

# In[10]:


# This image is from https://www.amazon.com/Hot-Toys-Marvel-Guardians-Life-Size/dp/B07257N92P
img = Image.open("img/baby-groot-toy.jpg") 

prompts = [
    "User: ",
    img,
#    "Who produced the movie that features this character?",
    "Who produced the movie that features this character? Think step-by-step.",
    "Assistant: ",
]

inputs = processor(prompts, return_tensors="pt").to(device)

generated_ids = model.generate(**inputs, max_length=100) #, eos_token_id=exit_condition, bad_words_ids=bad_words_ids)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

# ![](img/margherita-pizza.jpg)

# In[11]:


# This image is from https://eu.ooni.com/blogs/recipes/margherita-pizza

img = Image.open("img/margherita-pizza.jpg") 

prompts = [
    "User: ",
    img,
    "How do I make this? Think step by step.",
    "Assistant: ",
]

inputs = processor(prompts, return_tensors="pt").to(device)

generated_ids = model.generate(**inputs, max_length=1000, eos_token_id=exit_condition, bad_words_ids=bad_words_ids)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

# ![](img/nflx-5-year-stock-chart.png)

# In[12]:


img = Image.open("img/nflx-5-year-stock-chart.png") 

prompts = [
    "User: ",
    img,
    "Describe this image. Think step by step.",
    "Assistant: ",
]

inputs = processor(prompts, return_tensors="pt").to(device)

generated_ids = model.generate(**inputs, max_length=1000, eos_token_id=exit_condition, bad_words_ids=bad_words_ids)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

# # Not yet good at charts

# In[13]:


img = Image.open("img/nflx-5-year-stock-chart.png") 

prompts = [
    "User: ",
    img,
    "What is the maxmium stock price as shown in this chart. Think step by step.",
    "Assistant: ",
]

inputs = processor(prompts, return_tensors="pt").to(device)

generated_ids = model.generate(**inputs, max_length=1000, eos_token_id=exit_condition, bad_words_ids=bad_words_ids)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

# # Not yet good at charts

# In[14]:


img = Image.open("img/nflx-5-year-stock-chart.png") 

prompts = [
    "User: ",
    img,
    "What is the current stock price as shown in this chart. Think step by step.",
    "Assistant: ",
]

inputs = processor(prompts, return_tensors="pt").to(device)

generated_ids = model.generate(**inputs, max_length=1000, eos_token_id=exit_condition, bad_words_ids=bad_words_ids)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

# In[ ]:



