#!/usr/bin/env python
# coding: utf-8

# # Continued pre-training Llama 2 models on SageMaker JumpStart

# In[ ]:


# %pip install -U datasets==2.15.0

# In[ ]:


# %pip install -U --force-reinstall \
#              langchain==0.0.324 \
#              typing_extensions==4.7.1 \
#              pypdf==3.16.4

# ## Deploy Pre-trained Model
# 
# ---
# 
# First we will deploy the Llama-2 model as a SageMaker endpoint. To train/deploy 13B and 70B models, please change model_id to "meta-textgeneration-llama-2-7b" and "meta-textgeneration-llama-2-70b" respectively.
# 
# ---

# In[4]:


import sagemaker

sess = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()

# In[5]:


model_id, model_version = "meta-textgeneration-llama-2-7b", "2.*"

# In[6]:


from sagemaker.jumpstart.model import JumpStartModel

pretrained_model = JumpStartModel(model_id=model_id, model_version=model_version)
pretrained_predictor = pretrained_model.deploy()

# ## Invoke the endpoint
# 
# ---
# Next, we invoke the endpoint with some sample queries. Later, in this notebook, we will fine-tune this model with a custom dataset and carry out inference using the fine-tuned model. We will also show comparison between results obtained via the pre-trained and the fine-tuned models.
# 
# ---

# In[2]:


def print_response(payload, response):
    print(payload["inputs"])
    print(f"> {response[0]['generation']}")
    print("\n==================================\n")

# In[3]:


# payload = {
#     "inputsWhat is the size of the Amazon consumer business in 2022?e is",
#     "parameters": {
#         "max_new_token_p": 0.9,
#         "temperature": 0.6,
#         "return_full_text": False,
#     },
# }
# try:
#     response = pretrained_predictor.predict(payload, custom_attributes="accept_eula=true")
#     print_response(payload, response)
# except Exception as e:
#     print(e)

# ### Dataset formatting for continued pre-training
# 
# #### Domain adaptation fine-tuning
# The Text Generation model can also be fine-tuned on any domain specific dataset. After being fine-tuned on the domain specific dataset, the model
# is expected to generate domain specific text and solve various NLP tasks in that specific domain with **few shot prompting**.
# 
# Below are the instructions for how the training data should be formatted for input to the model.
# 
# - **Input:** A train and an optional validation directory. Each directory contains a CSV/JSON/TXT file. 
#   - For CSV/JSON files, the train or validation data is used from the column called 'text' or the first column if no column called 'text' is found.
#   - The number of files under train and validation (if provided) should equal to one, respectively. 
# - **Output:** A trained model that can be deployed for inference. 

# In[26]:


s3_prefix = f"s3://jumpstart-cache-prod-{boto3.Session().region_name}/training-datasets/sec_amazon"
s3_location = f"s3://jumpstart-cache-prod-{boto3.Session().region_name}/training-datasets/sec_amazon/AMZN_2021_2022_train_js.txt"

# In[27]:


!aws s3 cp $s3_location ./

# In[30]:


!head AMZN_2021_2022_train_js.txt

# In[15]:


import boto3
from sagemaker.jumpstart.estimator import JumpStartEstimator

model_id, model_version = "meta-textgeneration-llama-2-7b", "2.*"

estimator = JumpStartEstimator(model_id=model_id, 
                               model_version=model_version, 
                               environment={"accept_eula": "true"},
                               instance_type = "ml.g5.24xlarge")

estimator.set_hyperparameters(instruction_tuned="False", epoch="5")

estimator.fit({"training": s3_prefix})

# ### Deploy the continued pretrained model

# In[16]:


cpt_predictor = estimator.deploy()

# In[2]:


def print_response(payload, response):
    print(payload["inputs"])
    print(f"> {response[0]['generation']}")
    print("\n==================================\n")

# In[18]:


payload = {
    "inputs": "What is the size of the Amazon consumer business in 2022?",
    "parameters": {"max_new_tokens": 100},
}

try:
    response = cpt_predictor.predict(payload, custom_attributes="accept_eula=true")
    print_response(payload, response)
except Exception as e:
    print(e)

# In[19]:


# # Delete resources
# pretrained_predictor.delete_model()
# pretrained_predictor.delete_endpoint()
# cpt_predictor.delete_model()
# cpt_predictor.delete_endpoint()

# In[ ]:



