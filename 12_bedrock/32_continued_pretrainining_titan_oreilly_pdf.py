#!/usr/bin/env python
# coding: utf-8

# # Continued Pre-Training with Amazon Bedrock
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
# ### 3. Prepare the dataset for continued pre-training
# ### 4. Upload the dataset to S3
# ### 5. Customize the model with continued pre-training
# ### 6. Provision the custom model for inference
# ### 7. Test the custom model
# ### 8. Delete the provisioned model to save cost

# -----------

# ## 1. Setup

# In[3]:


%pip install -q -U --force-reinstall \
    boto3 \
    pandas==2.1.2 \
    langchain==0.0.324 \
    typing_extensions==4.7.1 \
    pypdf==3.16.4

# In[4]:


import boto3
import json
import time
from pprint import pprint
from IPython.display import display, HTML
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 2)
pd.set_option('display.max_colwidth', 1000)

# In[5]:


# Amazon Bedrock control plane including fine-tuning
bedrock = boto3.client(service_name="bedrock")

# Amazon Bedrock data plane including model inference
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# In[6]:


base_model_id = "amazon.titan-text-express-v1"

# # 2. Test the base model

# In[7]:


prompt = "Describe the book, 'Generative AI on AWS' by O'Reilly Media."

body = {
    "inputText": prompt,
    "textGenerationConfig": {
        "maxTokenCount": 512,
        "stopSequences": [],
        "temperature": 1,
        "topP": 0.9
    }
}

# ![](images/gaia_cover_sm.png)

# ### Amazon Titan Text model

# In[8]:


response = bedrock_runtime.invoke_model(
    modelId="amazon.titan-text-express-v1", # Amazon Titan Text model
    body=json.dumps(body)
)

response_body = response["body"].read().decode('utf8')
print(json.loads(response_body)["results"][0]["outputText"])

# # 3. Prepare the dataset for continued pre-training

# In[9]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/Generative_AI_on_AWS_OReilly.pdf")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 20000, # 4096 token context window * 6 chars per token ~= 24,576 
    chunk_overlap = 2000, # overlap for continuity across chunks
)

docs = text_splitter.split_documents(document)
docs[0]

# In[10]:


contents = ""
for doc in docs:
    content = {"input": doc.page_content}
    contents += (json.dumps(content) + "\n")

# In[11]:


with open("./train-continual-pretraining.jsonl", "w") as file:
    file.writelines(contents)
    file.close()

# In[12]:


import pandas as pd
df = pd.read_json("./train-continual-pretraining.jsonl", lines=True)
df

# In[13]:


data = "./train-continual-pretraining.jsonl"

# # 4. Upload dataset to S3

# Next, we need to upload our training dataset to S3:

# In[14]:


import sagemaker
sess = sagemaker.Session()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name
sagemaker_session_bucket = sess.default_bucket()

s3_location = f"s3://{sagemaker_session_bucket}/bedrock/finetuning/train-continual-pretraining.jsonl"
s3_output = f"s3://{sagemaker_session_bucket}/bedrock/finetuning/output"

# In[15]:


!aws s3 cp train-continual-pretraining.jsonl $s3_location

# # 5. Customize the model with continued pre-training

# In[16]:


timestamp = int(time.time())

job_name = "titan-{}".format(timestamp)
job_name

custom_model_name = "custom-{}".format(job_name)
custom_model_name

# In[17]:


bedrock.create_model_customization_job(
    customizationType="CONTINUED_PRE_TRAINING",
    jobName=job_name,
    customModelName=custom_model_name,
    roleArn=role,
    baseModelIdentifier=base_model_id,
    hyperParameters = {
        "epochCount": "10",
        "batchSize": "1",
        "learningRate": "0.000001"
    },
    trainingDataConfig={"s3Uri": s3_location},
    outputDataConfig={"s3Uri": s3_output},
)

# In[21]:


status = bedrock.get_model_customization_job(jobIdentifier=job_name)["status"]

while status == "InProgress":
    print(status)
    time.sleep(300)
    status = bedrock.get_model_customization_job(jobIdentifier=job_name)["status"]
    
print(status)

# In[22]:


custom_model_arn = bedrock.get_custom_model(modelIdentifier=custom_model_name)['modelArn']

display(
    HTML(
        '<b>Review <a target="blank" href="https://console.aws.amazon.com/bedrock/home?region={}#/custom-models/item?arn={}">Custom Model</a></b>'.format(
            region, custom_model_arn
        )
    )
)

# # 6. Provision the custom model for inference

# In[23]:


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

# In[24]:


deployment_status = bedrock.get_provisioned_model_throughput(
    provisionedModelId=provisioned_model_name)["status"]

while deployment_status == "Creating":    
    print(deployment_status)
    time.sleep(120)
    deployment_status = bedrock.get_provisioned_model_throughput(
        provisionedModelId=provisioned_model_name)["status"]  
    
print(deployment_status)

# In[25]:


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

# In[26]:


prompt = "Describe the book, 'Generative AI on AWS' by O'Reilly Media."

body = {
    "inputText": prompt,
    "textGenerationConfig": {
        "maxTokenCount": 512,
        "stopSequences": [],
        "temperature": 1,
        "topP": 0.9
    }
}

# ### Amazon Titan Text model

# In[27]:


response = bedrock_runtime.invoke_model(
    modelId="amazon.titan-text-express-v1", # Amazon Titan Text model
    body=json.dumps(body)
)

response_body = response["body"].read().decode('utf8')
print(json.loads(response_body)["results"][0]["outputText"])

# ### Our custom pre-trained model

# In[28]:


response = bedrock_runtime.invoke_model(
    modelId=provisioned_model_arn, # custom pre-trained model
    body=json.dumps(body)
)

response_body = response["body"].read().decode('utf8')
print(json.loads(response_body)["results"][0]["outputText"])

# # 8. Delete provisioned model to save cost

# In[29]:


bedrock.delete_provisioned_model_throughput(
    provisionedModelId = provisioned_model_name
)

# # GitHub repo
# https://github.com/generative-ai-on-aws/generative-ai-on-aws
# 
# ![](images/github.png)
