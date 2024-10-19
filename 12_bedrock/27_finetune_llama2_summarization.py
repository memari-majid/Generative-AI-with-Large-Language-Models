#!/usr/bin/env python
# coding: utf-8

# # Fine-Tuning Llama2 with Amazon Bedrock
# 
# #### GitHub repo
# https://github.com/generative-ai-on-aws/generative-ai-on-aws
# 
# ![](images/github.png)
# 
# _Note: This notebook was tested in Amazon SageMaker Studio with Python 3 (Data Science 3.0) kernel with the ml.t3.medium kernel._

# -----------

# ### 1. Setup
# ### 2. Test the base model
# ### 3. Prepare the dataset for fine-tuning
# ### 4. Upload the dataset to S3
# ### 5. Customize the model with fine-tuning
# ### 6. Provision the custom model for inference
# ### 7. Test the custom model
# ### 8. Delete the provisioned model to save cost

# ---

# # 1. Setup

# In[2]:


%pip install -q -U --force-reinstall \
    pandas==2.1.2 \
    datasets==2.15.0

# In[3]:


import boto3
import json
import time
from pprint import pprint
from IPython.display import display, HTML
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 2)
pd.set_option('display.max_colwidth', 1000)

# In[4]:


# Amazon Bedrock control plane including fine-tuning
bedrock = boto3.client(service_name="bedrock")

# Amazon Bedrock data plane including model inference
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# In[5]:


for model in bedrock.list_foundation_models(byProvider="meta", 
                                            byCustomizationType="FINE_TUNING")["modelSummaries"]:
    print("-----\n" + "modelArn: " + model["modelArn"] + "\nmodelId: " + model["modelId"] + "\nmodelName: " + model["modelName"] + "\ncustomizationsSupported: " + ','.join(model["customizationsSupported"]))

# In[6]:


base_model_id = "meta.llama2-13b-v1:0:4k"

# # 2. Test the base model

# In[7]:


prompt = """
Summarize the simplest and most interesting part of the following conversation.

#Person1#: Hello. My name is John Sandals, and I've got a reservation.
#Person2#: May I see some identification, sir, please?
#Person1#: Sure. Here you are.
#Person2#: Thank you so much. Have you got a credit card, Mr. Sandals?
#Person1#: I sure do. How about American Express?
#Person2#: Unfortunately, at the present time we take only MasterCard or VISA.
#Person1#: No American Express? Okay, here's my VISA.
#Person2#: Thank you, sir. You'll be in room 507, nonsmoking, with a queen-size bed. Do you approve, sir?
#Person1#: Yeah, that'll be fine.
#Person2#: That's great. This is your key, sir. If you need anything at all, anytime, just dial zero.

Summary: 
"""

body = {
    "prompt": prompt,
    "temperature": 0.5,
    "top_p": 0.9,
    "max_gen_len": 512,
}

# ### Llama2 chat model 

# In[7]:


response = bedrock_runtime.invoke_model(
    modelId="meta.llama2-13b-chat-v1", # compare to chat model
    body=json.dumps(body)
)

response_body = response["body"].read().decode('utf8')
print(json.loads(response_body)["generation"])

# # 3. Prepare the dataset
# https://huggingface.co/datasets/knkarthick/dialogsum
# 
# ![](images/dataset_sm.png)

# In[9]:


from datasets import load_dataset
dataset = load_dataset("knkarthick/dialogsum").remove_columns(["id", "topic"])
dataset

# In[10]:


pprint(dataset["train"][0])

# In[11]:


def wrap_instruction_fn(example):
    prompt = 'Summarize the simplest and most interesting part of the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    example["instruction"] = prompt + example["dialogue"] + end_prompt
    return example

# In[12]:


dataset['train']\
  .select(range(1000))\
  .select_columns(['dialogue', 'summary'])\
  .map(wrap_instruction_fn)\
  .remove_columns(['dialogue'])\
  .rename_column('instruction', 'prompt')\
  .rename_column('summary', 'completion')\
  .to_json('./train-summarization.jsonl', index=False)

# In[13]:


df = pd.read_json("./train-summarization.jsonl", lines=True)
df

# # 4. Upload our dataset to S3

# In[14]:


import sagemaker
sess = sagemaker.Session()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name
sagemaker_session_bucket = sess.default_bucket()

s3_location = f"s3://{sagemaker_session_bucket}/bedrock/finetuning/train-summarization.jsonl"
s3_output = f"s3://{sagemaker_session_bucket}/bedrock/finetuning/output"

# In[15]:


!aws s3 cp ./train-summarization.jsonl $s3_location

# # 5. Customize the model with fine-tuning

# In[16]:


timestamp = int(time.time())

job_name = "llama2-{}".format(timestamp)

custom_model_name = "custom-{}".format(job_name)
custom_model_name

# In[17]:


bedrock.create_model_customization_job(
    customizationType="FINE_TUNING",
    jobName=job_name,
    customModelName=custom_model_name,
    roleArn=role,
    baseModelIdentifier=base_model_id,
    hyperParameters = {
        "epochCount": "1",
        "batchSize": "1",
        "learningRate": "0.000005"
    },
    trainingDataConfig={"s3Uri": s3_location},
    outputDataConfig={"s3Uri": s3_output},
)

# In[18]:


status = bedrock.get_model_customization_job(jobIdentifier=job_name)["status"]

while status == "InProgress":
    print(status)
    time.sleep(600)
    status = bedrock.get_model_customization_job(jobIdentifier=job_name)["status"]
    
print(status)

# In[19]:


custom_model_arn = bedrock.get_custom_model(modelIdentifier=custom_model_name)['modelArn']

display(
    HTML(
        '<b>Review <a target="blank" href="https://console.aws.amazon.com/bedrock/home?region={}#/custom-models/item?arn={}">Custom Model</a></b>'.format(
            region, custom_model_arn
        )
    )
)

# # 6. Provision the custom model for inference

# In[20]:


provisioned_model_name = "{}-provisioned".format(custom_model_name)

base_model_arn = bedrock.get_custom_model(modelIdentifier=custom_model_name)['baseModelArn']

# Must do this manually through the console.  
# Use the value of "provisioned_model_name" for continuity.
#
# bedrock.create_provisioned_model_throughput(
#     modelUnits = 1,
#     provisionedModelName = provisioned_model_name,
#     modelId = base_model_arn
# ) 

# In[21]:


deployment_status = bedrock.get_provisioned_model_throughput(
    provisionedModelId=provisioned_model_name)["status"]

while deployment_status == "Creating":
    print(deployment_status)
    time.sleep(120)
    deployment_status = bedrock.get_provisioned_model_throughput(
        provisionedModelId=provisioned_model_name)["status"]  
    
print(deployment_status)

# In[22]:


provisioned_model_arn = bedrock.get_provisioned_model_throughput(
     provisionedModelId=provisioned_model_name)["provisionedModelArn"]

display(
    HTML(
        '<b>Review <a target="blank" href="https://console.aws.amazon.com/bedrock/home?region={}#/provisioned-throughput/details?arn={}">Custom Model Inference</a></b>'.format(
            region, provisioned_model_arn
        )
    )
)

# # 7. Test the custom model

# In[23]:


prompt = """
Summarize the simplest and most interesting part of the following conversation.

#Person1#: Hello. My name is John Sandals, and I've got a reservation.
#Person2#: May I see some identification, sir, please?
#Person1#: Sure. Here you are.
#Person2#: Thank you so much. Have you got a credit card, Mr. Sandals?
#Person1#: I sure do. How about American Express?
#Person2#: Unfortunately, at the present time we take only MasterCard or VISA.
#Person1#: No American Express? Okay, here's my VISA.
#Person2#: Thank you, sir. You'll be in room 507, nonsmoking, with a queen-size bed. Do you approve, sir?
#Person1#: Yeah, that'll be fine.\\n#Person2#: That's great. This is your key, sir. If you need anything at all, anytime, just dial zero.

Summary: 
"""

body = {
    "prompt": prompt,
    "temperature": 0.5,
    "top_p": 0.9,
    "max_gen_len": 512,
}

# ### Llama2 chat model 

# In[24]:


response = bedrock_runtime.invoke_model(
    modelId="meta.llama2-13b-chat-v1", # compare to chat model
    body=json.dumps(body)
)

response_body = response["body"].read().decode('utf8')
print(json.loads(response_body)["generation"])

# ### Our custom fine-tuned model

# In[25]:


response = bedrock_runtime.invoke_model(
    modelId=provisioned_model_arn, # custom fine-tuned model
    body=json.dumps(body)
)

response_body = response["body"].read().decode('utf8')
print(json.loads(response_body)["generation"])

# # 8. Delete provisioned model to save cost

# In[26]:


bedrock.delete_provisioned_model_throughput(
    provisionedModelId = provisioned_model_name
)

# # GitHub repo
# https://github.com/generative-ai-on-aws/generative-ai-on-aws
# 
# ![](images/github.png)
