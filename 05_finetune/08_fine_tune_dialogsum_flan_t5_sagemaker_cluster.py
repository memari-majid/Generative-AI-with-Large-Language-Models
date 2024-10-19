#!/usr/bin/env python
# coding: utf-8

# # Fine-tune the Instructor-Model for Dialogue Summarization using SageMaker Training Jobs
# 

# # NOTE:  THIS NOTEBOOK WILL TAKE ABOUT 20 MINUTES TO COMPLETE.
# 
# # PLEASE BE PATIENT.

# In[2]:


import boto3
import sagemaker
import pandas as pd

sess = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

import botocore.config

config = botocore.config.Config(
    user_agent_extra='gaia/1.0'
)

sm = boto3.Session().client(service_name="sagemaker", 
                            region_name=region, 
                            config=config)

# # _PRE-REQUISITE: You need to have succesfully run the notebooks in the `PREPARE` section before you continue with this notebook._

# In[3]:


%store -r processed_train_data_s3_uri

# In[4]:


try:
    processed_train_data_s3_uri
except NameError:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("[ERROR] Please run the notebooks in the PREPARE section before you continue.")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# In[5]:


print(processed_train_data_s3_uri)

# In[6]:


%store -r processed_validation_data_s3_uri

# In[7]:


try:
    processed_validation_data_s3_uri
except NameError:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("[ERROR] Please run the notebooks in the PREPARE section before you continue.")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# In[8]:


print(processed_validation_data_s3_uri)

# In[9]:


%store -r processed_test_data_s3_uri

# In[10]:


try:
    processed_test_data_s3_uri
except NameError:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("[ERROR] Please run the notebooks in the PREPARE section before you continue.")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# In[11]:


print(processed_test_data_s3_uri)

# # Specify the Dataset in S3
# We are using the train, validation, and test splits created in the previous section.

# In[12]:


print(processed_train_data_s3_uri)

!aws s3 ls $processed_train_data_s3_uri/

# In[13]:


print(processed_validation_data_s3_uri)

!aws s3 ls $processed_validation_data_s3_uri/

# In[14]:


print(processed_test_data_s3_uri)

!aws s3 ls $processed_test_data_s3_uri/

# # Specify S3 Input Data

# In[15]:


from sagemaker.inputs import TrainingInput

s3_input_train_data = TrainingInput(s3_data=processed_train_data_s3_uri)
s3_input_validation_data = TrainingInput(s3_data=processed_validation_data_s3_uri)
s3_input_test_data = TrainingInput(s3_data=processed_test_data_s3_uri)

print(s3_input_train_data.config)
print(s3_input_validation_data.config)
print(s3_input_test_data.config)

# # Setup Hyper-Parameters for FLAN Model

# In[16]:


model_checkpoint='google/flan-t5-base'

# In[17]:


epochs = 1 # increase this if you want to train for a longer period
learning_rate = 0.00001
weight_decay = 0.01
train_batch_size = 4
validation_batch_size = 4
test_batch_size = 4
train_instance_count = 1
train_instance_type = "ml.c5.9xlarge"
train_volume_size = 1024
input_mode = "FastFile"
train_sample_percentage = 0.01 # increase this if you want to train on more data

# # Setup Metrics To Track Model Performance

# In[18]:


metrics_definitions = [
    {"Name": "train:loss", "Regex": "'train_loss': ([0-9\\.]+)"},
    {"Name": "validation:loss", "Regex": "'eval_loss': ([0-9\\.]+)"},
]

# # Specify Checkpoint S3 Location
# This is used for Spot Instances Training.  If nodes are replaced, the new node will start training from the latest checkpoint.

# In[19]:


import uuid

checkpoint_s3_prefix = "checkpoints/{}".format(str(uuid.uuid4()))
checkpoint_s3_uri = "s3://{}/{}/".format(bucket, checkpoint_s3_prefix)

print(checkpoint_s3_uri)

# # Setup Our Script to Run on SageMaker
# Prepare our model to run on the managed SageMaker service

# In[20]:


!pygmentize src/train.py

# In[21]:


from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    role=role,
    instance_count=train_instance_count,
    instance_type=train_instance_type,
    volume_size=train_volume_size,
    checkpoint_s3_uri=checkpoint_s3_uri,
    py_version="py39",
    framework_version="1.13",
    hyperparameters={
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,        
        "train_batch_size": train_batch_size,
        "validation_batch_size": validation_batch_size,
        "test_batch_size": test_batch_size,
        "model_checkpoint": model_checkpoint,
        "train_sample_percentage": train_sample_percentage,
    },
    input_mode=input_mode,
    metric_definitions=metrics_definitions,
)

# # Train the Model on SageMaker

# In[22]:


estimator.fit(
    inputs={"train": s3_input_train_data, "validation": s3_input_validation_data, "test": s3_input_test_data},
    wait=False,
)

# In[23]:


training_job_name = estimator.latest_training_job.name
print("Training Job Name: {}".format(training_job_name))

# In[24]:


from IPython.core.display import display, HTML

display(
    HTML(
        '<b>Review <a target="blank" href="https://console.aws.amazon.com/sagemaker/home?region={}#/jobs/{}">Training Job</a> After About 5 Minutes</b>'.format(
            region, training_job_name
        )
    )
)

# In[25]:


from IPython.core.display import display, HTML

display(
    HTML(
        '<b>Review <a target="blank" href="https://console.aws.amazon.com/cloudwatch/home?region={}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={};streamFilter=typeLogStreamPrefix">CloudWatch Logs</a> After About 5 Minutes</b>'.format(
            region, training_job_name
        )
    )
)

# In[26]:


from IPython.core.display import display, HTML

display(
    HTML(
        '<b>Review <a target="blank" href="https://s3.console.aws.amazon.com/s3/buckets/{}/{}/?region={}&tab=overview">S3 Output Data</a> After The Training Job Has Completed</b>'.format(
            bucket, training_job_name, region
        )
    )
)

# In[27]:


%%time

estimator.latest_training_job.wait(logs=False)

# # Deploy the Fine-Tuned Model to a Real Time Endpoint

# In[28]:


sm_model = estimator.create_model(
    entry_point='inference.py',
    source_dir='src',
)
endpoint_name = training_job_name.replace('pytorch-training', 'summary-tuned')
predictor = sm_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.2xlarge',
    endpoint_name=endpoint_name
)

# # Zero Shot Inference with the Fine-Tuned Model in a SageMaker Endpoint

# In[29]:


zero_shot_prompt = """Summarize the following conversation.

#Person1#: Tom, I've got good news for you.
#Person2#: What is it?
#Person1#: Haven't you heard that your novel has won The Nobel Prize?
#Person2#: Really? I can't believe it. It's like a dream come true. I never expected that I would win The Nobel Prize!
#Person1#: You did a good job. I'm extremely proud of you.
#Person2#: Thanks for the compliment.
#Person1#: You certainly deserve it. Let's celebrate!

Summary:"""

# In[30]:


import json
from sagemaker import Predictor
predictor = Predictor(
    endpoint_name=endpoint_name,
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

# # Tear Down the Endpoint

# In[31]:


# predictor.delete_endpoint()
