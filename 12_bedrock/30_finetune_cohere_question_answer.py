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


%pip install -U --force-reinstall pandas==2.1.2 datasets==2.14.6

# In[3]:


# %pip install -U boto3-1.28.60-py3-none-any.whl \
#                botocore-1.31.60-py3-none-any.whl \
#                awscli-1.29.60-py3-none-any.whl \
#                --force-reinstall --quiet

# In[4]:


%pip list | grep boto3

# #### Now let's set up our connection to the Amazon Bedrock SDK using Boto3

# In[5]:


#### Un comment the following lines to run from your local environment outside of the AWS account with Bedrock access

#import os
#os.environ['BEDROCK_ASSUME_ROLE'] = '<YOUR_VALUES>'
#os.environ['AWS_PROFILE'] = '<YOUR_VALUES>'

# In[6]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# In[7]:


bedrock.list_foundation_models()

# In[8]:


import sagemaker

sess = sagemaker.Session()
sagemaker_session_bucket = sess.default_bucket()
role = sagemaker.get_execution_role()

# ### Invoke Model before Fine-Training

# In[9]:


command_base_model_id = "cohere.command-light-text-v14"

# In[10]:


base_model_id = "cohere.command-light-text-v14:7:4k"

# ### Convert the dataset into jsonlines format

# In[11]:


from datasets import load_dataset

dataset = load_dataset("csv", data_files="gaia_questions_answers.csv")

# In[12]:


print(dataset)

# In[13]:


def wrap_instruction_fn(example):
    prompt = 'Answer the following question:\n\n'
    end_prompt = '\n\nAnswer: '
    example["instruction"] = prompt + example["question"] + end_prompt
    return example

# In[14]:


dataset['train']\
  .filter(lambda example: example['question'] and example['answer'])\
  .select_columns(['question', 'answer'])\
  .map(wrap_instruction_fn)\
  .rename_column('instruction', 'prompt')\
  .rename_column('answer', 'completion')\
  .remove_columns(['question'])\
  .to_json('./train-question-answer.jsonl', index=False)

# dataset['validation']\
#   .filter(lambda example: example['question'])\
#   .select_columns(['question', 'answer'])\
#   .map(wrap_instruction_fn)\
#   .rename_column('instruction', 'input')\
#   .rename_column('answer', 'output')\
#   .to_json('./validation-summarization.jsonl', index=False)

# dataset['test']\
#   .filter(lambda example: example['question'])\
#   .select_columns(['question', 'answer'])\
#   .map(wrap_instruction_fn)\
#   .rename_column('instruction', 'input')\
#   .rename_column('answer', 'output')\
#   .to_json('./test-summarization.jsonl', index=False)

#  .remove_columns(['Unnamed: 0.1', 'Unnamed: 0', 'question_id', 'title', 'answer_id', 'expertreviewed', 'upvotedcount', 'tags'])\

# In[15]:


import pandas as pd
df = pd.read_json("./train-question-answer.jsonl", lines=True)
df

# In[16]:


data = "./train-question-answer.jsonl"

# Read the JSON line file into an object like any normal file

# In[17]:


with open(data) as f:
    lines = f.read().splitlines()

# #### Load the ‘lines’ object into a pandas Data Frame.

# In[18]:


import pandas as pd
df_inter = pd.DataFrame(lines)
df_inter.columns = ['json_element']

# This intermediate data frame will have only one column with each json object in a row. A sample output is given below.

# In[19]:


df_inter['json_element'].apply(json.loads)

# Now we will apply json loads function on each row of the ‘json_element’ column. ‘json.loads’ is a decoder function in python which is used to decode a json object into a dictionary. ‘apply’ is a popular function in pandas that takes any function and applies to each row of the pandas dataframe or series.

# In[20]:


df_final = pd.json_normalize(df_inter['json_element'].apply(json.loads))

# Once decoding is done we will apply the json normalize function to the above result. json normalize will convert any semi-structured json data into a flat table. Here it converts the JSON ‘keys’ to columns and its corresponding values to row elements.

# In[21]:


df_final

# ### Uploading data to S3

# Next, we need to upload our training dataset to S3:

# In[22]:


s3_location = f"s3://{sagemaker_session_bucket}/bedrock/finetuning/train-question-answer.jsonl"
s3_output = f"s3://{sagemaker_session_bucket}/bedrock/finetuning/output"

# In[23]:


!aws s3 cp ./train-question-answer.jsonl $s3_location

# Now we can create the fine-tuning job. 
# 
# ### ^^ **Note:** Make sure the IAM role you're using has these [IAM policies](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-iam-role.html) attached that allow Amazon Bedrock access to the specified S3 buckets ^^

# ## 2. Fine-tuning

# In[24]:


import time
timestamp = int(time.time())

# In[25]:


job_name = "cohere-{}".format(timestamp)
job_name

# In[26]:


custom_model_name = "custom-{}".format(job_name)
custom_model_name

# In[27]:


bedrock.create_model_customization_job(
    jobName=job_name,
    customModelName=custom_model_name,
    roleArn=role,
    baseModelIdentifier=base_model_id,
    hyperParameters = {
        "epochCount": "10",
        "batchSize": "8",
        "learningRate": "0.000005",
    },
    trainingDataConfig={"s3Uri": s3_location},
    outputDataConfig={"s3Uri": s3_output},
)

# In[28]:


status = bedrock.get_model_customization_job(jobIdentifier=job_name)["status"]
status

# # Let's periodically check in on the progress.
# ### The next cell might run for ~40min

# In[29]:


import time

status = bedrock.get_model_customization_job(jobIdentifier=job_name)["status"]

while status == "InProgress":
    print(status)
    time.sleep(30)
    status = bedrock.get_model_customization_job(jobIdentifier=job_name)["status"]
    
print(status)

# In[30]:


completed_job = bedrock.get_model_customization_job(jobIdentifier=job_name)
completed_job

# ## 3. Testing

# Now we can test the fine-tuned model

# In[31]:


bedrock.list_custom_models()

# In[32]:


for job in bedrock.list_model_customization_jobs()["modelCustomizationJobSummaries"]:
    print("-----\n" + "jobArn: " + job["jobArn"] + "\njobName: " + job["jobName"] + "\nstatus: " + job["status"] + "\ncustomModelName: " + job["customModelName"])

# ## GetCustomModel

# In[33]:


bedrock.get_custom_model(modelIdentifier=custom_model_name)

# In[ ]:


custom_model_arn = bedrock.get_custom_model(modelIdentifier=custom_model_name)['modelArn']
custom_model_arn

# In[ ]:


base_model_arn = bedrock.get_custom_model(modelIdentifier=custom_model_name)['baseModelArn']
base_model_arn

# ## **Note:** To invoke custom models, you need to first create a provisioned throughput resource and make requests using that resource.

# In[ ]:


provisioned_model_name = "{}-provisioned".format(custom_model_name)
provisioned_model_name

# ## !! **Note:** SDK currently only supports 1 month and 6 months commitment terms. Go to Bedrock console to manually purchase no commitment term option for testing !!

# In[ ]:


# bedrock.create_provisioned_model_throughput(
#     modelUnits = 1,
#     commitmentDuration = "OneMonth", ## Note: SDK is currently missing No Commitment option
#     provisionedModelName = provisioned_model_name,
#     modelId = base_model_arn
# ) 

# ## ListProvisionedModelThroughputs

# In[ ]:


bedrock.list_provisioned_model_throughputs()["provisionedModelSummaries"]

# ## GetProvisionedModelThroughput

# In[ ]:


#provisioned_model_name = "<YOUR_PROVISIONED_MODEL_NAME>" # e.g. custom-titan-1698257909-provisioned
#provisioned_model_name = "custom-titan-1700411048-provisioned"

# In[ ]:


provisioned_model_arn = bedrock.get_provisioned_model_throughput(
     provisionedModelId=provisioned_model_name)["provisionedModelArn"]
provisioned_model_arn

# In[ ]:


deployment_status = bedrock.get_provisioned_model_throughput(
    provisionedModelId=provisioned_model_name)["status"]
deployment_status

# ## The next cell might run for ~10min

# In[ ]:


import time

deployment_status = bedrock.get_provisioned_model_throughput(
    provisionedModelId=provisioned_model_name)["status"]

while deployment_status == "Creating":
    
    print(deployment_status)
    time.sleep(30)
    deployment_status = bedrock.get_provisioned_model_throughput(
        provisionedModelId=provisioned_model_name)["status"]  
    
print(deployment_status)

# # Qualitative Results with Zero Shot Inference BEFORE and AFTER Fine-Tuning
# 
# As with many GenAI applications, a qualitative approach where you ask yourself the question "is my model behaving the way it is supposed to?" is usually a good starting point. In the example below (the same one we started this notebook with), you can see how the fine-tuned model is able to create a reasonable summary of the dialogue compared to the original inability to understand what is being asked of the model.

# ## Before Fine-Tuning

# In[ ]:


command_base_model_id = "cohere.command-light-text-v14"

# In[ ]:


prompt = "Answer the following question:\n\nHow can prompt engineering go wrong?\n\nAnswer: "

# In[ ]:


body = {
    "prompt": prompt,
    "temperature": 0.5,
    "top_p": 0.9,
    "max_gen_len": 512,
}

response = bedrock_runtime.invoke_model(
    modelId=command_base_model_id, 
    body=json.dumps(body)
)

response_body = response["body"].read().decode('utf8')
print(json.loads(response_body)["generation"])

# ## After Fine-Tuning

# In[ ]:


body = {
    "prompt": prompt,
    "temperature": 0.5,
    "top_p": 0.9,
    "max_gen_len": 512,
}

response = bedrock_runtime.invoke_model(
    modelId=provisioned_model_arn,
    body=json.dumps(body)
)

response_body = response["body"].read().decode('utf8')
print(json.loads(response_body)["generation"])

# ## Delete Provisioned Throughput

# When you're done testing, you can delete Provisioned Throughput to stop charges

# In[ ]:


# bedrock.delete_provisioned_model_throughput(
#     provisionedModelId = provisioned_model_name
# )
