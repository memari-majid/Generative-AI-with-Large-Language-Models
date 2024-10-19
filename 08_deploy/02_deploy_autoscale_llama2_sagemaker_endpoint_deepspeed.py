#!/usr/bin/env python
# coding: utf-8

# # Deploy LLama2 70b Model with high performance on SageMaker using Sagemaker LMI and Rolling batch
# 

# 
# In this notebook, we explore how to host a LLama2 large language model with FP16 precision on SageMaker using the DeepSpeed. We use DJLServing as the model serving solution in this example that is bundled in the LMI container. DJLServing is a high-performance universal model serving solution powered by the Deep Java Library (DJL) that is programming language agnostic. To learn more about DJL and DJLServing, you can refer to our recent blog post (https://aws.amazon.com/blogs/machine-learning/deploy-bloom-176b-and-opt-30b-on-amazon-sagemaker-with-large-model-inference-deep-learning-containers-and-deepspeed/).
# 
# 
# Model parallelism can help deploy large models that would normally be too large for a single GPU. With model parallelism, we partition and distribute a model across multiple GPUs. Each GPU holds a different part of the model, resolving the memory capacity issue for the largest deep learning models with billions of parameters. 
# 
# SageMaker has rolled out DeepSpeed container which now provides users with the ability to leverage the managed serving capabilities and help to provide the un-differentiated heavy lifting.
# 
# In this notebook, we deploy https://huggingface.co/TheBloke/Llama-2-70b-fp16 model on a ml.g5.48xlarge instance. 

# # Licence agreement
#  - View license information https://huggingface.co/meta-llama before using the model.
#  - This notebook is a sample notebook and not intended for production use. Please refer to the licence at https://github.com/aws/mit-0. 

# In[2]:


# !pip install sagemaker boto3 --upgrade

# In[3]:


import sagemaker
import jinja2
from sagemaker import image_uris
import boto3
import os
import time
import json
from pathlib import Path

# In[4]:


role = sagemaker.get_execution_role()  # execution role for the endpoint
sess = sagemaker.session.Session()  # sagemaker session for interacting with different AWS APIs
bucket = sess.default_bucket()  # bucket to house artifacts

# In[23]:


model_bucket = sess.default_bucket()  # bucket to house model artifacts
s3_code_prefix = "hf-large-model-djl/meta-llama/Llama-2-70b-fp16/code"  # folder within bucket where code artifact will go

s3_model_prefix = "hf-large-model-djl/meta-llama/Llama-2-70b-fp16/model"  # folder within bucket where model artifact will go
region = sess._region_name
account_id = sess.account_id()

s3_client = boto3.client("s3")
sagemaker_client = boto3.client("sagemaker")
sagemaker_runtime_client = boto3.client("sagemaker-runtime")

jinja_env = jinja2.Environment()

# ### Define a variable to contain the s3url of the location that has the model

# In[9]:


# Define a variable to contain the s3url of the location that has the model. For demo purpose, we use Llama-2-70b-fp16 model artifacts from our S3 bucket
pretrained_model_location = f"s3://sagemaker-example-files-prod-{region}/models/llama-2/fp16/70B/"

# ## Create SageMaker compatible Model artifact,  upload Model to S3 and bring your own inference script.
# 
# SageMaker Large Model Inference containers can be used to host models without providing your own inference code. This is extremely useful when there is no custom pre-processing of the input data or postprocessing of the model's predictions.
# 
# SageMaker needs the model artifacts to be in a Tarball format. In this example, we provide the following files - serving.properties.
# 
# The tarball is in the following format:
# 
# ```
# code
# ├──── 
# │   └── serving.properties
# ```
# 
#     serving.properties is the configuration file that can be used to configure the model server.
# 

# #### Create serving.properties 
# This is a configuration file to indicate to DJL Serving which model parallelization and inference optimization libraries you would like to use. Depending on your need, you can set the appropriate configuration.
# 
# Here is a list of settings that we use in this configuration file -
# 
#     engine: The engine for DJL to use. In this case, we have set it to MPI.
#     option.model_id: The model id of a pretrained model hosted inside a model repository on huggingface.co (https://huggingface.co/models) or S3 path to the model artifacts. 
#     option.tensor_parallel_degree: Set to the number of GPU devices over which Accelerate needs to partition the model. This parameter also controls the no of workers per model which will be started up when DJL serving runs. As an example if we have a 4 GPU machine and we are creating 4 partitions then we will have 1 worker per model to serve the requests.
# 
# For more details on the configuration options and an exhaustive list, you can refer the documentation - https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-large-model-configuration.html.
# 
# 

# In[10]:


!rm -rf code_llama2_70b_fp16
!mkdir -p code_llama2_70b_fp16

# In[11]:


%%writefile code_llama2_70b_fp16/serving.properties
engine = MPI
option.tensor_parallel_degree = 8
option.rolling_batch = auto
option.max_rolling_batch_size = 4
option.model_loading_timeout = 3600
option.model_id = {{model_id}}
option.paged_attention = true
option.trust_remote_code = true
option.dtype = fp16

# In[12]:


# we plug in the appropriate model location into our `serving.properties`
template = jinja_env.from_string(Path("code_llama2_70b_fp16/serving.properties").open().read())
Path("code_llama2_70b_fp16/serving.properties").open("w").write(
    template.render(model_id=pretrained_model_location)
)
!pygmentize code_llama2_70b_fp16/serving.properties | cat -n

# **Image URI for the DJL container is being used here**

# In[13]:


inference_image_uri = image_uris.retrieve(
    framework="djl-deepspeed", region=region, version="0.23.0"
)
print(f"Image going to be used is ---- > {inference_image_uri}")

# **Create the Tarball and then upload to S3 location**

# In[14]:


!rm model.tar.gz
!tar czvf model.tar.gz code_llama2_70b_fp16

# In[15]:


s3_code_artifact = sess.upload_data("model.tar.gz", bucket, s3_code_prefix)

# ### To create the end point the steps are:
# 
# 1. Create the Model using the Image container and the Model Tarball uploaded earlier
# 2. Create the endpoint config using the following key parameters
# 
#     a) Instance Type is ml.g5.48xlarge 
#     
#     b) ContainerStartupHealthCheckTimeoutInSeconds is 3600 to ensure health check starts after the model is ready    
# 3. Create the end point using the endpoint config created    
# 

# #### Create the Model
# Use the image URI for the DJL container and the s3 location to which the tarball was uploaded.
# 
# The container downloads the model into the `/tmp` space on the instance because SageMaker maps the `/tmp` to the Amazon Elastic Block Store (Amazon EBS) volume that is mounted when we specify the endpoint creation parameter VolumeSizeInGB. 
# It leverages `s5cmd`(https://github.com/peak/s5cmd) which offers a very fast download speed and hence extremely useful when downloading large models.
# 
# For instances like p4dn, which come pre-built with the volume instance, we can continue to leverage the `/tmp` on the container. The size of this mount is large enough to hold the model.
# 

# In[16]:


from sagemaker.utils import name_from_base

model_name = name_from_base(f"Llama-2-70b-fp16-mpi")
print(model_name)

create_model_response = sagemaker_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer={
        "Image": inference_image_uri,
        "ModelDataUrl": s3_code_artifact,
        "Environment": {"MODEL_LOADING_TIMEOUT": "3600"},
    },
)
model_arn = create_model_response["ModelArn"]

print(f"Created Model: {model_arn}")

# In[17]:


endpoint_config_name = f"{model_name}-config"
endpoint_name = f"{model_name}-endpoint"

endpoint_config_response = sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "variant1",
            "ModelName": model_name,
            "InstanceType": "ml.g5.48xlarge",
            "InitialInstanceCount": 1,
            "ModelDataDownloadTimeoutInSeconds": 3600,
            "ContainerStartupHealthCheckTimeoutInSeconds": 3600,
        },
    ],
)
endpoint_config_response

# In[19]:


create_endpoint_response = sagemaker_client.create_endpoint(
    EndpointName=f"{endpoint_name}", EndpointConfigName=endpoint_config_name
)
print(f"Created Endpoint: {create_endpoint_response['EndpointArn']}")

# In[20]:


from IPython.core.display import display, HTML

display(
    HTML(
        '<b>Review <a target="blank" href="https://console.aws.amazon.com/sagemaker/home?region={}#/endpoints/{}">SageMaker REST Endpoint</a></b>'.format(
            region, endpoint_name
        )
    )
)

# ### This step can take ~ 20 min or longer so please be patient

# In[21]:


import time

resp = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
status = resp["EndpointStatus"]
print("Status: " + status)

while status == "Creating":
    time.sleep(60)
    resp = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    print("Status: " + status)

print("Arn: " + resp["EndpointArn"])
print("Status: " + status)

# #### While you wait for the endpoint to be created, you can read more about:
# - [Deep Learning containers for large model inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-large-model-dlc.html)

# #### Leverage the Boto3 to invoke the endpoint. 
# 
# This is a generative model so we pass in a Text as a prompt and Model will complete the sentence and return the results.
# 
# You can pass a prompt as input to the model. This done by setting inputs to a prompt. The model then returns a result for each prompt. The text generation can be configured using appropriate parameters.
# These parameters need to be passed to the endpoint as a dictionary of kwargs. Refer this documentation - https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig for more details.
# 
# The below code sample illustrates the invocation of the endpoint using a text prompt and also sets some parameters

# In[ ]:


sagemaker_runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps(
        {
            "inputs": "The diamondback terrapin was the first reptile to do what?",
            "parameters": {
                "do_sample": True,
                "max_new_tokens": 100,
                "min_new_tokens": 100,
                "temperature": 0.3,
                "watermark": True,
            },
        }
    ),
    ContentType="application/json",
)["Body"].read().decode("utf8")

# # Autoscaling a SageMaker Endpoint

# In[29]:


autoscale = boto3.Session().client(service_name="application-autoscaling")

# In[31]:


autoscale.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/" + endpoint_name + "/variant/variant1",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=2,
    RoleARN=role,
    SuspendedState={
        "DynamicScalingInSuspended": False,
        "DynamicScalingOutSuspended": False,
        "ScheduledScalingSuspended": False,
    },
)

# In[32]:


# check the target is available
autoscale.describe_scalable_targets(
    ServiceNamespace="sagemaker",
    MaxResults=100,
)

# In[33]:


autoscale.put_scaling_policy(
    PolicyName="autoscale-policy-gpu-400-llama2-70b",
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/" + endpoint_name + "/variant/variant1",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 400, # 400% of 800% total GPU utilization (8 GPUs)
        "CustomizedMetricSpecification":
        {
            "MetricName": "GPUUtilization",
            "Namespace": "/aws/sagemaker/Endpoints",
            "Dimensions": [
                {"Name": "EndpointName", "Value": endpoint_name},
                {"Name": "VariantName", "Value": "variant1"}
            ],
            "Statistic": "Average",
            "Unit": "Percent"
        },
        "ScaleOutCooldown": 60,
        "ScaleInCooldown": 300,
    }
)

# # Trigger the Autoscaling

# In[ ]:


for i in range(0, 100):
    res = sagemaker_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            {
                "inputs": "The diamondback terrapin was the first reptile to do what?",
                "parameters": {
                    "do_sample": True,
                    "max_new_tokens": 100,
                    "min_new_tokens": 100,
                    "temperature": 0.3,
                    "watermark": True,
                },
            }
        ),
        ContentType="application/json",
    )
    print(f'{i}: {res["Body"].read().decode("utf8")}')

# In[37]:


autoscale.describe_scaling_activities(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/" + endpoint_name + "/variant/variant1",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MaxResults=100
)

# ## Clean Up

# In[ ]:


# # - Delete the end point
# sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

# In[ ]:


# # - In case the end point failed we still want to delete the model
# sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
# sagemaker_client.delete_model(ModelName=model_name)
