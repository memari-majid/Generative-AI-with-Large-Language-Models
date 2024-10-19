#!/usr/bin/env python
# coding: utf-8

# # Deploy Mixtral 8x7B on Amazon SageMaker
# 
# _Note: This notebook is derived from this notebook:  https://github.com/philschmid/llm-sagemaker-sample_
# 
# [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) is the open LLM from [Mistral AI](https://huggingface.co/mistralai). The Mixtral-8x7B is a Sparse Mixture of Experts it has a similar architecture to Mistral 7B, but comes with a twist: it’s actually 8 “expert” models in one. If you want to learn more about MoEs check out [Mixture of Experts Explained](https://huggingface.co/blog/moe). 
# 
# In this blog you will learn how to deploy [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model to Amazon SageMaker. We are going to use the Hugging Face LLM DLC is a new purpose-built Inference Container to easily deploy LLMs in a secure and managed environment. The DLC is powered by [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) a scalelable, optimized solution for deploying and serving Large Language Models (LLMs). The Blog post also includes Hardware requirements for the different model sizes. 
# 
# In the blog will cover how to:
# 1. [Setup development environment](#1-setup-development-environment)
# 2. [Retrieve the new Hugging Face LLM DLC](#2-retrieve-the-new-hugging-face-llm-dlc)
# 3. [Hardware requirements](#3-hardware-requirements)
# 4. [Deploy Mixtral 8x7B to Amazon SageMaker](#4-deploy-mixtral-8x7b-to-amazon-sagemaker)
# 5. [Run inference and chat with the model](#5-run-inference-and-chat-with-the-model)
# 6. [Clean up](#5-clean-up)
# 
# Lets get started!
# 

# ## 1. Setup development environment
# 
# We are going to use the `sagemaker` python SDK to deploy Mixtral to Amazon SageMaker. We need to make sure to have an AWS account configured and the `sagemaker` python SDK installed. 

# In[4]:


import sagemaker
import boto3
sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker session region: {sess.boto_region_name}")


# ## 2. Retrieve the new Hugging Face LLM DLC
# 
# Compared to deploying regular Hugging Face models we first need to retrieve the container uri and provide it to our `HuggingFaceModel` model class with a `image_uri` pointing to the image. To retrieve the new Hugging Face LLM DLC in Amazon SageMaker, we can use the `get_huggingface_llm_image_uri` method provided by the `sagemaker` SDK. This method allows us to retrieve the URI for the desired Hugging Face LLM DLC based on the specified `backend`, `session`, `region`, and `version`. You can find the available versions [here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-text-generation-inference-containers)
# 
# _Note: At the time of writing this blog post the latest version of the Hugging Face LLM DLC is not yet available via the `get_huggingface_llm_image_uri` method. We are going to use the raw container uri instead._
# 

# In[5]:


# COMMENT IN WHEN PR (https://github.com/aws/sagemaker-python-sdk/pull/4314) IS MERGED
from sagemaker.huggingface import get_huggingface_llm_image_uri

# NEEDS 1.3.3 which is not yet available - even in the PR mentioned above
# # retrieve the llm image uri
# llm_image = get_huggingface_llm_image_uri(
#   "huggingface",
#   version="1.3.3"
# )

region_mapping = {
    "af-south-1": "626614931356",
    "il-central-1": "780543022126",
    "ap-east-1": "871362719292",
    "ap-northeast-1": "763104351884",
    "ap-northeast-2": "763104351884",
    "ap-northeast-3": "364406365360",
    "ap-south-1": "763104351884",
    "ap-south-2": "772153158452",
    "ap-southeast-1": "763104351884",
    "ap-southeast-2": "763104351884",
    "ap-southeast-3": "907027046896",
    "ap-southeast-4": "457447274322",
    "ca-central-1": "763104351884",
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
    "eu-central-1": "763104351884",
    "eu-central-2": "380420809688",
    "eu-north-1": "763104351884",
    "eu-west-1": "763104351884",
    "eu-west-2": "763104351884",
    "eu-west-3": "763104351884",
    "eu-south-1": "692866216735",
    "eu-south-2": "503227376785",
    "me-south-1": "217643126080",
    "me-central-1": "914824155844",
    "sa-east-1": "763104351884",
    "us-east-1": "763104351884",
    "us-east-2": "763104351884",
    "us-gov-east-1": "446045086412",
    "us-gov-west-1": "442386744353",
    "us-iso-east-1": "886529160074",
    "us-isob-east-1": "094389454867",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
}

# Changed from 1.3.1 to 1.3.3
llm_image = f"{region_mapping[sess.boto_region_name]}.dkr.ecr.{sess.boto_region_name}.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.3.3-gpu-py310-cu121-ubuntu20.04-v1.0"

# print ecr image uri
print(f"llm image uri: {llm_image}")

# ## 3. Hardware requirements
# 
# [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) is 45B parameter Mixture-of-Experts (MoE) model. This means we’ll need to have enough VRAM to hold a dense 47B parameter model. Why 47B parameters and not 8 x 7B = 56B? That’s because in MoE models, only the FFN layers are treated as individual experts. This allows Mixtral 8x7B to achieve the latency of ~12B models by being 45B parameters. 
# 
# > Note: The Hugging Face LLM DLC version 1.3.1 does not yet support quantization techniques with AWQ and GPTQ. We expect to support these techniques in the next releases. We will update this blog post once the new version is available. 
# 
# | Model                                                                       | Instance Type       | Quantization | NUM_GPUS | 
# |-----------------------------------------------------------------------------|---------------------|--------------|----------|
# | [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | `(ml.)p4d.24xlarge` | `-`          | 8        | 
# | [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | `(ml.)g5.48xlarge`  | `-`          | 8        | 
# 
# 
# The parameter tested for `MAX_INPUT_LENGTH`, `MAX_BATCH_PREFILL_TOKENS`, `MAX_TOTAL_TOKENS` and `MAX_BATCH_TOTAL_TOKENS` are the same for both instances. You can tweak them for your needs. 

# ## 4. Deploy Mixtral 8x7B to Amazon SageMaker
# 
# To deploy [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) to Amazon SageMaker we create a `HuggingFaceModel` model class and define our endpoint configuration including the `hf_model_id`, `instance_type` etc. We will use a `g5.45xlarge` instance type, which has 8 NVIDIA A10G GPUs and 192GB of GPU memory. You need atleast > 100GB of GPU memory to run Mixtral 8x7B in float16 with decent input length. 

# In[6]:


import json
from sagemaker.huggingface import HuggingFaceModel

# sagemaker config
instance_type = "ml.g5.48xlarge"
number_of_gpu = 8
health_check_timeout = 300

# Define Model and Endpoint configuration parameter
config = {
  'HF_MODEL_ID': "mistralai/Mixtral-8x7B-Instruct-v0.1", # model_id from hf.co/models
  'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
  'MAX_INPUT_LENGTH': json.dumps(24000),  # Max length of input text
  'MAX_BATCH_PREFILL_TOKENS': json.dumps(32000),  # Number of tokens for the prefill operation.
  'MAX_TOTAL_TOKENS': json.dumps(32000),  # Max length of the generation (including input text)
  'MAX_BATCH_TOTAL_TOKENS': json.dumps(512000),  # Limits the number of tokens that can be processed in parallel during the generation
  # ,'HF_MODEL_QUANTIZE': "awq", # comment in to quantize not supported yet
}

# create HuggingFaceModel with the image uri
llm_model = HuggingFaceModel(
  role=role,
  image_uri=llm_image,
  env=config
)

# After we have created the `HuggingFaceModel` we can deploy it to Amazon SageMaker using the `deploy` method. We will deploy the model with the `ml.g5.48xlarge` instance type. TGI will automatically distribute and shard the model across all GPUs.

# In[7]:


# Deploy model to an endpoint
# https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy
llm = llm_model.deploy(
  initial_instance_count=1,
  instance_type=instance_type,
  container_startup_health_check_timeout=health_check_timeout, # 10 minutes to be able to load the model
)


# SageMaker will now create our endpoint and deploy the model to it. This can takes a 10-15 minutes. 

# ## 5. Run inference and chat with the model
# 
# After our endpoint is deployed we can run inference on it. We will use the `predict` method from the `predictor` to run inference on our endpoint. We can inference with different parameters to impact the generation. Parameters can be defined as in the `parameters` attribute of the payload. You can find supported parameters in the [here](https://www.philschmid.de/sagemaker-llama-llm#5-run-inference-and-chat-with-the-model) or in the open api specification of the TGI in the [swagger documentation](https://huggingface.github.io/text-generation-inference/)
# 
# The `mistralai/Mixtral-8x7B-Instruct-v0.1` is a conversational chat model meaning we can chat with it using the following prompt:
#   
# ```
# <s> [INST] User Instruction 1 [/INST] Model answer 1</s> [INST] User instruction 2 [/INST]
# ```
# Lets see, if Mixtral can come up with some cool ideas for the summer.

# In[8]:


# Prompt to generate
prompt=f"<s> [INST] What are cool ideas for the summer to do? List 5. [/INST] "

# Generation arguments
payload = {
    "do_sample": True,
    "top_p": 0.6,
    "temperature": 0.9,
    "top_k": 50,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.03,
    "return_full_text": False,
    "stop": ["</s>"]
}

# Okay lets test it.

# In[9]:


chat = llm.predict({"inputs":prompt, "parameters":payload})

print(chat[0]["generated_text"])

# ## 6. Clean up
# 
# To clean up, we can delete the model and endpoint.
# 

# In[ ]:


# llm.delete_model()
# llm.delete_endpoint()
