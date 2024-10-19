#!/usr/bin/env python
# coding: utf-8

# # Approve and Deploy the Model
# 
# The pipeline that was executed created a Model Package version within the specified Model Package Group. Of particular note, the registration of the model/creation of the Model Package was done so with approval status as `PendingManualApproval`.
# 
# As part of SageMaker Pipelines, data scientists can register the model with approved/pending manual approval as part of the CI/CD workflow.
# 
# We can also approve the model using the SageMaker Studio UI or programmatically as shown below.

# In[2]:


from botocore.exceptions import ClientError

import os
import sagemaker
import logging
import boto3
import sagemaker

sess = sagemaker.Session()
bucket = sess.default_bucket()
region = boto3.Session().region_name
role = sagemaker.get_execution_role()

import botocore.config

config = botocore.config.Config(
    user_agent_extra='gaia/1.0'
)

sm = boto3.Session().client(service_name="sagemaker", 
                            region_name=region,
                            config=config)

# # List Pipeline Execution Steps
# 

# In[3]:


%store -r pipeline_name

# In[4]:


print(pipeline_name)

# In[5]:


%%time

import time
from pprint import pprint

executions_response = sm.list_pipeline_executions(PipelineName=pipeline_name)["PipelineExecutionSummaries"]
pipeline_execution_status = executions_response[0]["PipelineExecutionStatus"]
print(pipeline_execution_status)

while pipeline_execution_status == "Executing":
    try:
        executions_response = sm.list_pipeline_executions(PipelineName=pipeline_name)["PipelineExecutionSummaries"]
        pipeline_execution_status = executions_response[0]["PipelineExecutionStatus"]
    except Exception as e:
        print("Please wait...")
        time.sleep(30)

pprint(executions_response)

# In[6]:


pipeline_execution_status = executions_response[0]["PipelineExecutionStatus"]
print(pipeline_execution_status)

# In[7]:


pipeline_execution_arn = executions_response[0]["PipelineExecutionArn"]
print(pipeline_execution_arn)

# In[8]:


from pprint import pprint

steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=pipeline_execution_arn)

pprint(steps)

# # View Registered Model

# In[9]:


for execution_step in steps["PipelineExecutionSteps"]:
    if execution_step["StepName"] == "Summarization-RegisterModel":
        model_package_arn = execution_step["Metadata"]["RegisterModel"]["Arn"]
        break
print(model_package_arn)

# # Approve the Model for Deployment

# In[10]:


model_package_update_response = sm.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Approved",  # Other options are Rejected and PendingManualApproval
)

# # View Created Model

# In[11]:


for execution_step in steps["PipelineExecutionSteps"]:
    if execution_step["StepName"] == "CreateModel":
        model_arn = execution_step["Metadata"]["Model"]["Arn"]
        break
print(model_arn)

pipeline_model_name = model_arn.split("/")[-1]
print(pipeline_model_name)

# # Create Model Endpoint from Model Registry
# More details here:  https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-deploy.html
# 

# In[12]:


import time

timestamp = int(time.time())

model_from_registry_name = "model-from-registry-{}".format(timestamp)
print("Model from registry name : {}".format(model_from_registry_name))

model_registry_package_container = {
    "ModelPackageName": model_package_arn,
}

# In[13]:


from pprint import pprint

create_model_from_registry_response = sm.create_model(
    ModelName=model_from_registry_name, ExecutionRoleArn=role, PrimaryContainer=model_registry_package_container
)
pprint(create_model_from_registry_response)

# In[14]:


model_from_registry_arn = create_model_from_registry_response["ModelArn"]
model_from_registry_arn

# #### Note: In this workshop, we are intentionally deploying our model to only 1 instance. The general recommendation is to deploy to 2 or more instances for automatic placement across two AZs for high availability.

# In[15]:


endpoint_config_name = "model-from-registry-epc-{}".format(timestamp)
print(endpoint_config_name)

create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "InstanceType": "ml.m5.4xlarge",
            "InitialVariantWeight": 1,
            "InitialInstanceCount": 1,
            "ModelName": pipeline_model_name,
            "VariantName": "AllTraffic",
        }
    ],
)

# In[16]:


%store -d pipeline_endpoint_name

# In[17]:


# Pick up the existing `pipeline_endpoint_name` if it was already created
%store -r pipeline_endpoint_name

# In[18]:


timestamp = int(time.time())
pipeline_endpoint_name = "model-from-registry-ep-{}".format(timestamp)
print("Created Pipeline EndpointName={}".format(pipeline_endpoint_name))

create_endpoint_response = sm.create_endpoint(
    EndpointName=pipeline_endpoint_name, EndpointConfigName=endpoint_config_name
)
print(create_endpoint_response["EndpointArn"])

# In[19]:


%store pipeline_endpoint_name

# In[20]:


from IPython.core.display import display, HTML

display(
    HTML(
        '<b>Review <a target="blank" href="https://console.aws.amazon.com/sagemaker/home?region={}#/endpoints/{}">SageMaker HTTPS Endpoint</a></b>'.format(
            region, pipeline_endpoint_name
        )
    )
)

# # _Wait Until the Endpoint is Deployed_
# _Note:  This will take a few minutes.  Please be patient._

# In[21]:


%%time

waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(EndpointName=pipeline_endpoint_name)
