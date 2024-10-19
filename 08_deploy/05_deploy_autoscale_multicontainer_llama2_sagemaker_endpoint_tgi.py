#!/usr/bin/env python
# coding: utf-8

# # Scale LLM Inference on Amazon SageMaker with Multi-Replica Endpoints
# 
# This notebook is derived from this blog post:  https://www.philschmid.de/sagemaker-multi-replica
# 
# One of the key Amazon SageMaker announcements at this year's re:Invent (2023) was the new Hardware Requirements object for Amazon SageMaker endpoints. This provides granular control over the compute resources for models deployed on SageMaker, including minimum CPU, GPU, memory, and number of replicas. This allows you to optimize your model's throughput and cost by matching the compute resources to the model's requirements and allows you to deploy multiple LLMs on the same instance. Previously it was not possible to deploy multiple replicas of a LLM or multiple LLMs on a single endpoint,  can limit the overall throughput of models are not compute bound, e.g. open LLMs like a single Llama 13B on p4d.24xlarge instances. 
# 
# In this post, we show how to use the new feature using the SageMaker SDK and `ResourceRequirements` object to optimize the deployment of Llama 2 for increased throughput and cost performance on Amazon SageMaker.
# 
# The instance we use here has 8x GPUs, which allows us to deploy 8 replicas of Llama 2 on a single instance. You can also use this example to deploy other open LLMs like Mistral, T5 or StarCoder. Additionally it is possible to deploy multiple models on a single instance, e.g. 4x Llama 13B and 4x Mistral 7B. Check out the amazing [blog post from Antje for this](https://aws.amazon.com/de/blogs/aws/amazon-sagemaker-adds-new-inference-capabilities-to-help-reduce-foundation-model-deployment-costs-and-latency/). 
# 
# We are going to use the Hugging Face LLM DLC is a new purpose-built Inference Container to easily deploy LLMs in a secure and managed environment. The DLC is powered by [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) a scalelable, optimized solution for deploying and serving Large Language Models (LLMs).

# ## Setup development environment
# 
# We are going to use the `sagemaker` python SDK to deploy Llama 2 to Amazon SageMaker. We need to make sure to have an AWS account configured and the `sagemaker` python SDK installed. 

# In[ ]:


# %pip install -U sagemaker==2.203.1 transformers==4.35.2 torch==2.1.0

# If you are going to use Sagemaker in a local environment. You need access to an IAM Role with the required permissions for Sagemaker. You can find [here](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) more about it.
# 

# In[25]:


import sagemaker
import boto3
sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

region = sess._region_name

role = sagemaker.get_execution_role()
role_name = role[role.rindex("/")+1:]

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker session region: {sess.boto_region_name}")
print(f"sagemaker role name: {role_name}")

# ## Retrieve the new Hugging Face LLM DLC
# 
# Compared to deploying regular Hugging Face models we first need to retrieve the container uri and provide it to our `HuggingFaceModel` model class with a `image_uri` pointing to the image. To retrieve the new Hugging Face LLM DLC in Amazon SageMaker, we can use the `get_huggingface_llm_image_uri` method provided by the `sagemaker` SDK. This method allows us to retrieve the URI for the desired Hugging Face LLM DLC based on the specified `backend`, `session`, `region`, and `version`. You can find the available versions [here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-text-generation-inference-containers)
# 

# In[4]:


from sagemaker.huggingface import get_huggingface_llm_image_uri

# retrieve the llm image uri
llm_image = get_huggingface_llm_image_uri(
  "huggingface",
  version="1.3.3"
)

# print ecr image uri
print(f"llm image uri: {llm_image}")

# ## 3. Configure Hardware requirements per replica
# 
# Llama 2 comes in 3 different sizes - 7B, 13B & 70B parameters. The hardware requirements will vary based on the model size deployed to SageMaker. Below is an example configuration for Llama 13B. In addition we tried to provide some high level overview of the different hardware requirements for the different model sizes. To keep it simple we only looked at the `p4d.24xlarge` instance type and AWQ/GPTQ quantization. 
# 
# | Model                                                              | Instance Type       | Quantization | # replica |
# |--------------------------------------------------------------------|---------------------|--------------|-----------|
# | [Llama 7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)   | `(ml.)p4d.24xlarge` | `-`          | 8         |
# | [Llama 7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)   | `(ml.)p4d.24xlarge` | `GPTQ/AWQ`   | 8         |
# | [Llama 13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | `(ml.)p4d.24xlarge` | `-`          | 8         |
# | [Llama 13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | `(ml.)p4d.24xlarge` | `GPTQ/AWQ`   | 8         |
# | [Llama 70B](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) | `(ml.)p4d.24xlarge` | `-`          | 2         |
# | [Llama 70B](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) | `(ml.)p4d.24xlarge` | `GPTQ/AWQ`   | 4         |

# `ResourceRequirements` configures multiple SageMaker `InferenceComponent` containers to run with a single SageMaker `Endpoint`. This provides more deployment flexibility than a single-model endpoint. 
# 
# In addition, multi-container deployments on a single endpoint separates the operations of the endpoint startup from the container startup so the overhead of endpoint startup is only incurred once.

# In[5]:


from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements

llama2_resource_config = ResourceRequirements(
    requests = {
        "copies": 8, # Number of replicas
        "num_accelerators": 1, # Number of GPUs
        "num_cpus": 10,  # Number of CPU cores 192 // num_replica - more for management
        "memory": 50 * 1024,  # Minimum memory (MB) 768 // num_replica - more for management
    },
)

# ## Deploy Llama 2 to Amazon SageMaker
# 
# To deploy the model to Amazon SageMaker we create a `HuggingFaceModel` model class and define our endpoint configuration including the `hf_model_id`, `instance_type` and then add our `ResourceRequirements` object to the `deploy` method. 
# 
# _Note: This is a form to enable access to Llama 2 on Hugging Face after you have been granted access from Meta. Please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept our license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days. We alternatively use the ungated weights from `NousResearch`._

# In[6]:


import json
import uuid
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.enums import EndpointType

# sagemaker config
instance_type = "ml.g5.48xlarge"
health_check_timeout = 300

# Define Model and Endpoint configuration parameter
config = {
  'HF_MODEL_ID': "NousResearch/Llama-2-7b-chat-hf", # "meta-llama/Llama-2-7b-chat-hf", # model_id from hf.co/models
  'SM_NUM_GPUS': json.dumps(1), # Number of GPU used per replica
  'MAX_INPUT_LENGTH': json.dumps(2048),  # Max length of input text
  'MAX_TOTAL_TOKENS': json.dumps(4096),  # Max length of the generation (including input text)
  'MAX_BATCH_TOTAL_TOKENS': json.dumps(16384),  # Limits the number of tokens that can be processed in parallel during the generation
  #'HF_MODEL_QUANTIZE': "gptq", # comment in when using awq quantized checkpoint

}

# create HuggingFaceModel with the image uri
llm_model = HuggingFaceModel(
  role=role,
  image_uri=llm_image,
  env=config,
)

# ### Deploy the model
# After we have created the `HuggingFaceModel` we can deploy it to Amazon SageMaker using the `deploy` method using the `ResourceRequirements` object. 
# 
# _Note: The particular `ResourceRequirements` configuration we are using may take ~20 mins since we are requesting that a single endpoint loading 8 replicas/containers.  This is because each endpoint deploys a container serially._
# 
# Ways to make it faster are:
# * Deploy the 8 replicas across >1 endpoints to reduce the number of replicas per endpoint.
# * Deploy 1 replica initially on the single instance (using a single GPU as configured in `ResourceRequirements`) and then later scale up to 8 replicas. This results in a single copy of the container that is available to serve traffic as quickly as possible.  Then the other 7 replicas will spin up separately so that the single endpoint is not waiting for all 8 replicas to startup.

# In[7]:


%%time

# Deploy model to an endpoint
# https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy
llm = llm_model.deploy(
  initial_instance_count=1, # number of instances
  instance_type=instance_type, # base instance type
  resources=llama2_resource_config, # resource config for multi-replica
  container_startup_health_check_timeout=health_check_timeout, # 10 minutes to be able to load the model
  endpoint_name=f"llama2-chat-{str(uuid.uuid4())}", # name needs to be unique
  endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED, # needed to use resource config
  tags=[{"Key": "aKey", "Value": "aValue"}],
  model_name="llama2-chat"
)

# In[36]:


inference_component_name = llm_model.sagemaker_session.list_inference_components(endpoint_name_equals=llm.endpoint_name).get("InferenceComponents")[0].get("InferenceComponentName")

endpoint_name = llm.endpoint_name

# In[ ]:


from IPython.core.display import display, HTML

display(
    HTML(
        '<b>Review <a target="blank" href="https://console.aws.amazon.com/sagemaker/home?region={}#/endpoints/{}">SageMaker Endpoint</a></b>'.format(
            region, endpoint_name
        )
    )
)

# SageMaker will now create our endpoint and deploy the model to it. This can takes a 15-25 minutes, since the replicas are deployed after each other. After the endpoint is created we can use the `predict` method to send a request to our endpoint. To make it easier we will use the [apply_chat_template](apply_chat_template) method from transformers. This allow us to send "openai" like converstaions to our model. 

# In[8]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(config["HF_MODEL_ID"])

# Conversational messages
messages = [
  {"role": "system", "content": "You are an helpful AWS Expert Assistant. Respond only with 1-2 sentences."},
  {"role": "user", "content": "What is Amazon SageMaker?"},
]

# generation parameters
parameters = {
    "do_sample" : True,
    "top_p": 0.6,
    "temperature": 0.9,
    "top_k": 50,
    "max_new_tokens": 50,
    "repetition_penalty": 1.03,
    "return_full_text": False,
}

res = llm.predict(
  {
    "inputs": tokenizer.apply_chat_template(messages, tokenize=False),
    "parameters": parameters
   })

print(res[0]['generated_text'].strip())

# # Autoscaling a SageMaker Endpoint

# In[26]:


autoscale = boto3.Session().client(service_name="application-autoscaling")

# In[38]:


autoscale.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/" + endpoint_name + "/variant/AllTraffic",
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

# In[39]:


# check the target is available
autoscale.describe_scalable_targets(
    ServiceNamespace="sagemaker",
    MaxResults=100,
)

# In[40]:


autoscale.put_scaling_policy(
    PolicyName="autoscale-policy-gpu-400-llama2-7b",
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/" + endpoint_name + "/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 400, # 400% of 800% total GPU utilization (8 GPUs)
        "CustomizedMetricSpecification":
        {
            "MetricName": "GPUUtilization",
            "Namespace": "/aws/sagemaker/Endpoints",
            "Dimensions": [
                {"Name": "EndpointName", "Value": endpoint_name },
                {"Name": "VariantName", "Value": "AllTraffic"}
            ],
            "Statistic": "Average",
            "Unit": "Percent"
        },
        "ScaleOutCooldown": 60,
        "ScaleInCooldown": 300,
    }
)

# ## Trigger autoscaling

# In[ ]:


for i in range(0, 100):
    res = llm.predict(
      {
        "inputs": tokenizer.apply_chat_template(messages, tokenize=False),
        "parameters": parameters
       })

    print(f"{i}: {res[0]['generated_text'].strip()}")

# In[42]:


autoscale.describe_scaling_activities(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/" + endpoint_name + "/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MaxResults=100
)

# ## Clean up
# 
# To clean up, we can delete the model, endpoint and inference component for the hardware requirements. 
# 
# _Note: If you have issues deleting an endpoint with an attached inference component, see: https://repost.aws/es/questions/QUEiuS2we2TEKe9GUUYm67kQ/error-when-deleting-and-inference-endpoint-in-sagemaker_

# In[16]:


# # Delete Inference Component & Model
# llm_model.sagemaker_session.delete_inference_component(inference_component_name=inference_component)
# llm.delete_model()

# ### We have to wait until the component is deleted before we can delete the endpoint. (can take 2minutes)

# In[17]:


# import time
# time.sleep(120)

# # If this call fails, you can delete the endpoint manually using the AWS Console
# llm.delete_endpoint()

# ### Ignore the rest of this

# In[46]:


# import boto3

# sagemaker_client = boto3.client("sagemaker")
# sagemaker_runtime_client = boto3.client("sagemaker-runtime")

# In[53]:


# endpoint_name = "llama2-chat-8a47b40a-e335-4c50-9bd8-0720b69f9efc"

# components = sagemaker_client.list_inference_components(
#     EndpointNameEquals=endpoint_name,
# )["InferenceComponents"]

# for component in components:
#     print(component)
#     sagemaker_client.delete_inference_component(InferenceComponentName=component['InferenceComponentName'])

# In[ ]:



