#!/usr/bin/env python
# coding: utf-8

# # Deploy the Model
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

import botocore.config

config = botocore.config.Config(
    user_agent_extra='gaia/1.0'
)

sm = boto3.Session().client(service_name="sagemaker", 
                            region_name=region,
                            config=config)

# # Retrieve model endpoint
# 

# In[3]:


%store -r pipeline_endpoint_name

# In[4]:


try:
    pipeline_endpoint_name
except NameError:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("[ERROR] Please run previous notebooks before you continue.")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# In[5]:


print(pipeline_endpoint_name)

# In[6]:


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

# In[7]:


%%time

waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(EndpointName=pipeline_endpoint_name)

# # _Wait Until the Endpoint ^^ Above ^^ is Deployed_

# # Zero Shot Inference

# In[8]:


import json
from sagemaker import Predictor

zero_shot_prompt = """Summarize the following conversation.

#Person1#: Tom, I've got good news for you.
#Person2#: What is it?
#Person1#: Haven't you heard that your novel has won The Nobel Prize?
#Person2#: Really? I can't believe it. It's like a dream come true. I never expected that I would win The Nobel Prize!
#Person1#: You did a good job. I'm extremely proud of you.
#Person2#: Thanks for the compliment.
#Person1#: You certainly deserve it. Let's celebrate!

Summary:"""
predictor = Predictor(
    endpoint_name=pipeline_endpoint_name,
    sagemaker_session=sess,
)
response = predictor.predict(zero_shot_prompt,
        {
            "ContentType": "application/x-text",
            "Accept": "application/json",
        },
)
response_json = json.loads(response.decode('utf-8'))
print(response_json)

# # Clean Up: Tear Down Endpoint

# In[ ]:


# sm.delete_endpoint(
#     EndpointName=pipeline_endpoint_name
# )
