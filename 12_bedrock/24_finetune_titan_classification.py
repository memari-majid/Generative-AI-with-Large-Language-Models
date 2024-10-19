#!/usr/bin/env python
# coding: utf-8

# # Introduction to Bedrock - Fine-Tuning
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 

# In this demo notebook, we demonstrate how to use the Bedrock Python SDK for fine-tuning Bedrock models with your own data. If you have text samples to train and want to adapt the Bedrock models to your domain, you can further fine-tune the Bedrock foundation models by providing your own training datasets. You can upload your datasets to Amazon S3, and provide the S3 bucket path while configuring a Bedrock fine-tuning job. You can also adjust hyper parameters (learning rate, epoch, and batch size) for fine-tuning. After the fine-tuning job of the model with your dataset has completed, you can start using the model for inference in the Bedrock playground application. You can select the fine-tuned model and submit a prompt to the fine-tuned model along with a set of model parameters. The fine-tuned model should generate texts to be more alike your text samples. 

# -----------

# 1. Setup
# 2. Fine-tuning
# 3. Testing the fine-tuned model

#  Note: This notebook was tested in Amazon SageMaker Studio with Python 3 (Data Science 2.0) kernel.

# ---

# ## 1. Setup

# In[2]:


%pip install -U boto3 botocore --force-reinstall --quiet

# In[3]:


%pip list | grep boto3

# #### Now let's set up our connection to the Amazon Bedrock SDK using Boto3

# In[4]:


#### Un comment the following lines to run from your local environment outside of the AWS account with Bedrock access

#import os
#os.environ['BEDROCK_ASSUME_ROLE'] = '<YOUR_VALUES>'
#os.environ['AWS_PROFILE'] = '<YOUR_VALUES>'

# In[2]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# In[3]:


bedrock_runtime.list_foundation_models()

# In[4]:


import sagemaker

sess = sagemaker.Session()
sagemaker_session_bucket = sess.default_bucket()
role = sagemaker.get_execution_role()

# ### Preview custom data

# Our custom data in JSON line file format.

# In[8]:


data = "data/train.jsonl"

# Read the JSON line file into an object like any normal file

# In[9]:


with open(data) as f:
    lines = f.read().splitlines()

# #### Load the ‘lines’ object into a pandas Data Frame.

# In[10]:


import pandas as pd
df_inter = pd.DataFrame(lines)
df_inter.columns = ['json_element']

# This intermediate data frame will have only one column with each json object in a row. A sample output is given below.

# In[11]:


df_inter['json_element'].apply(json.loads)

# Now we will apply json loads function on each row of the ‘json_element’ column. ‘json.loads’ is a decoder function in python which is used to decode a json object into a dictionary. ‘apply’ is a popular function in pandas that takes any function and applies to each row of the pandas dataframe or series.

# In[12]:


df_final = pd.json_normalize(df_inter['json_element'].apply(json.loads))

# Once decoding is done we will apply the json normalize function to the above result. json normalize will convert any semi-structured json data into a flat table. Here it converts the JSON ‘keys’ to columns and its corresponding values to row elements.

# In[13]:


df_final

# ### Uploading data to S3

# Next, we need to upload our training dataset to S3:

# In[14]:


s3_location = f"s3://{sagemaker_session_bucket}/bedrock/finetuning/train.jsonl"
s3_output = f"s3://{sagemaker_session_bucket}/bedrock/finetuning/output"

# In[15]:


!aws s3 cp data/train.jsonl $s3_location

# Now we can create the fine-tuning job. 
# 
# ### ^^ **Note:** Make sure the IAM role you're using has these [IAM policies](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-iam-role.html) attached that allow Amazon Bedrock access to the specified S3 buckets ^^

# ## 2. Fine-tuning

# In[16]:


import time
timestamp = int(time.time())

# In[17]:


base_model_id = "amazon.titan-text-express-v1"
#base_model_id = "amazon.titan-text-lite-v1"

# In[18]:


job_name = "titan-{}".format(timestamp)
job_name

# In[19]:


custom_model_name = "custom-{}".format(job_name)
custom_model_name

# In[21]:


bedrock_runtime.create_model_customization_job(
    customizationType="FINE_TUNING",
    jobName=job_name,
    customModelName=custom_model_name,
    roleArn=role,
    baseModelIdentifier=base_model_id,
    hyperParameters = {
        "epochCount": "1",
        "batchSize": "1",
        "learningRate": "0.005",
        "learningRateWarmupSteps": "0"
    },
    trainingDataConfig={"s3Uri": s3_location},
    outputDataConfig={"s3Uri": s3_output},
)

# In[22]:


status = bedrock_runtime.get_model_customization_job(jobIdentifier=job_name)["status"]
status

# # Let's periodically check in on the progress.
# ### The next cell might run for ~40min

# In[23]:


import time

status = bedrock_runtime.get_model_customization_job(jobIdentifier=job_name)["status"]

while status == "InProgress":
    print(status)
    time.sleep(30)
    status = bedrock_runtime.get_model_customization_job(jobIdentifier=job_name)["status"]
    
print(status)

# In[24]:


completed_job = bedrock_runtime.get_model_customization_job(jobIdentifier=job_name)
completed_job

# ## 3. Testing

# Now we can test the fine-tuned model

# In[25]:


bedrock_runtime.list_custom_models()

# In[26]:


for job in bedrock_runtime.list_model_customization_jobs()["modelCustomizationJobSummaries"]:
    print("-----\n" + "jobArn: " + job["jobArn"] + "\njobName: " + job["jobName"] + "\nstatus: " + job["status"] + "\ncustomModelName: " + job["customModelName"])

# ## GetCustomModel

# In[27]:


bedrock_runtime.get_custom_model(modelIdentifier=custom_model_name)

# In[28]:


custom_model_arn = bedrock_runtime.get_custom_model(modelIdentifier=custom_model_name)['modelArn']
custom_model_arn

# In[29]:


base_model_arn = bedrock_runtime.get_custom_model(modelIdentifier=custom_model_name)['baseModelArn']
base_model_arn

# ## **Note:** To invoke custom models, you need to first create a provisioned throughput resource and make requests using that resource.

# In[30]:


provisioned_model_name = "{}-provisioned".format(custom_model_name)
provisioned_model_name

# ## !! **Note:** SDK currently only supports 1 month and 6 months commitment terms. Go to Bedrock console to manually purchase no commitment term option for testing !!

# In[ ]:


# bedrock_runtime.create_provisioned_model_throughput(
#     modelUnits = 1,
#     commitmentDuration = "OneMonth", ## Note: SDK is currently missing No Commitment option
#     provisionedModelName = provisioned_model_name,
#     modelId = base_model_arn
# ) 

# ## ListProvisionedModelThroughputs

# In[ ]:


bedrock_runtime.list_provisioned_model_throughputs()["provisionedModelSummaries"]

# ## GetProvisionedModelThroughput

# In[5]:


#provisioned_model_name = "<YOUR_PROVISIONED_MODEL_NAME>" # e.g. custom-titan-1698257909-provisioned
#provisioned_model_name = "custom-titan-1698257909-provisioned" 

# In[6]:


provisioned_model_arn = bedrock_runtime.get_provisioned_model_throughput(
     provisionedModelId=provisioned_model_name)["provisionedModelArn"]
provisioned_model_arn

# In[7]:


deployment_status = bedrock_runtime.get_provisioned_model_throughput(
    provisionedModelId=provisioned_model_name)["status"]
deployment_status

# ## The next cell might run for ~10min

# In[8]:


import time

deployment_status = bedrock_runtime.get_provisioned_model_throughput(
    provisionedModelId=provisioned_model_name)["status"]

while deployment_status == "Creating":
    
    print(deployment_status)
    time.sleep(30)
    deployment_status = bedrock_runtime.get_provisioned_model_throughput(
        provisionedModelId=provisioned_model_name)["status"]  
    
print(deployment_status)

# # Qualitative Results with Zero Shot Inference AFTER Fine-Tuning
# 
# As with many GenAI applications, a qualitative approach where you ask yourself the question "is my model behaving the way it is supposed to?" is usually a good starting point. In the example below (the same one we started this notebook with), you can see how the fine-tuned model is able to create a reasonable summary of the dialogue compared to the original inability to understand what is being asked of the model.

# In[9]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# In[18]:


response = bedrock_runtime.invoke_model(
    # modelId needs to be Provisioned Throughput Model ARN
    modelId=provisioned_model_arn,
    body="""
{
  "inputText": "Classify this statement as Positive, Neutral, or Negative:\\n'I really do not like this!'",
  "textGenerationConfig":{
    "maxTokenCount": 1, 
    "stopSequences": [],
    "temperature": 1,
    "topP": 0.9
  }
}
"""
)

response_body = response["body"].read().decode('utf8')
print(response_body)

print(json.loads(response_body)["results"][0]["outputText"])

# In[17]:


response = bedrock_runtime.invoke_model(
    # modelId needs to be Provisioned Throughput Model ARN
    modelId=provisioned_model_arn,
    body="""
{
  "inputText": "Classify this statement as Positive, Neutral, or Negative:\\n'I really like this!'",
  "textGenerationConfig":{
    "maxTokenCount": 1, 
    "stopSequences": [],
    "temperature": 1,
    "topP": 0.9
  }
}
"""
)

response_body = response["body"].read().decode('utf8')
print(response_body)

print(json.loads(response_body)["results"][0]["outputText"])

# In[ ]:



