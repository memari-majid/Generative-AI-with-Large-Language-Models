#!/usr/bin/env python
# coding: utf-8

# # Feature Transformation with Amazon a SageMaker Processing Job
# 
# Typically a machine learning (ML) process consists of few steps. First, gathering data with various ETL jobs, then pre-processing the data, featurizing the dataset by incorporating standard techniques or prior knowledge, and finally training an ML model using an algorithm.
# 
# Often, distributed data processing frameworks such as Scikit-Learn, Spark, Ray, and others are used to pre-process data sets in order to prepare them for training. In this notebook we'll use Amazon SageMaker Processing, and leverage the power of HuggingFace in a managed SageMaker environment to run our processing workload.

# # NOTE:  THIS NOTEBOOK WILL TAKE A 5-10 MINUTES TO COMPLETE.
# 
# # PLEASE BE PATIENT.

# ## Contents
# 
# 1. Setup Environment
# 1. Setup Input Data
# 1. Setup Output Data
# 1. Build a Scikit-Learn container for running the processing job
# 1. Run the Processing Job using Amazon SageMaker
# 1. Inspect the Processed Output Data

# # Setup Environment
# 
# Let's start by specifying:
# * The S3 bucket and prefixes that you use for training and model data. Use the default bucket specified by the Amazon SageMaker session.
# * The IAM role ARN used to give processing and training access to the dataset.

# In[2]:


import sagemaker
import boto3

sess = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sess.default_bucket()
region = boto3.Session().region_name

import botocore.config

config = botocore.config.Config(
    user_agent_extra='gaia/1.0'
)

sm = boto3.Session().client(service_name="sagemaker", 
                            region_name=region, 
                            config=config)
s3 = boto3.Session().client(service_name="s3", 
                            region_name=region,
                            config=config)

# # Setup Input Data S3 URI

# In[3]:


sess.upload_data('./data-summarization/dialogsum-1.csv', bucket=bucket, key_prefix='data-summarization')
sess.upload_data('./data-summarization/dialogsum-2.csv', bucket=bucket, key_prefix='data-summarization')

# In[4]:


raw_input_data_s3_uri = f's3://{bucket}/data-summarization/'
print(raw_input_data_s3_uri)

# In[5]:


!aws s3 ls $raw_input_data_s3_uri

# In[6]:


 

# # Run the Processing Job using Amazon SageMaker
# 
# Next, use the Amazon SageMaker Python SDK to submit a processing job using our custom python script.

# # Review the Processing Script

# In[7]:


!pygmentize preprocess.py

# Run this script as a processing job.  You also need to specify one `ProcessingInput` with the `source` argument of the Amazon S3 bucket and `destination` is where the script reads this data from `/opt/ml/processing/input` (inside the Docker container.)  All local paths inside the processing container must begin with `/opt/ml/processing/`.
# 
# Also give the `run()` method a `ProcessingOutput`, where the `source` is the path the script writes output data to.  For outputs, the `destination` defaults to an S3 bucket that the Amazon SageMaker Python SDK creates for you, following the format `s3://sagemaker-<region>-<account_id>/<processing_job_name>/output/<output_name>/`.  You also give the `ProcessingOutput` value for `output_name`, to make it easier to retrieve these output artifacts after the job is run.
# 
# The arguments parameter in the `run()` method are command-line arguments in our `preprocess.py` script.
# 
# Note that we sharding the data using `ShardedByS3Key` to spread the transformations across all worker nodes in the cluster.

# In[8]:


processing_instance_type = "ml.c5.2xlarge"
processing_instance_count = 2
train_split_percentage = 0.9
validation_split_percentage = 0.05
test_split_percentage = 0.05

# In[9]:


from sagemaker.sklearn.processing import SKLearnProcessor

processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    env={"AWS_DEFAULT_REGION": region},
    max_runtime_in_seconds=7200,
)

# In[10]:


input_s3 = f's3://{bucket}/data-summarization/'

# In[11]:


!aws s3 ls {input_s3}

# In[12]:


from sagemaker.processing import ProcessingInput, ProcessingOutput

processor.run(
    code="preprocess.py",
    inputs=[
        ProcessingInput(
            input_name="raw-input-data",
            source=raw_input_data_s3_uri,
            destination="/opt/ml/processing/input/data/",
            s3_data_distribution_type="ShardedByS3Key",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train", 
            s3_upload_mode="EndOfJob", 
            source="/opt/ml/processing/output/data/train"
        ),
        ProcessingOutput(
            output_name="validation",
            s3_upload_mode="EndOfJob",
            source="/opt/ml/processing/output/data/validation",
        ),
        ProcessingOutput(
            output_name="test", 
            s3_upload_mode="EndOfJob", 
            source="/opt/ml/processing/output/data/test"
        ),
    ],
    arguments=[
        "--train-split-percentage",
        str(train_split_percentage),
        "--validation-split-percentage",
        str(validation_split_percentage),
        "--test-split-percentage",
        str(test_split_percentage),
        "--model-checkpoint",
        str(model_checkpoint),
    ],
    logs=True,
    wait=False,
)

# In[13]:


scikit_processing_job_name = processor.jobs[-1].describe()["ProcessingJobName"]
print(scikit_processing_job_name)

# In[14]:


from IPython.core.display import display, HTML

display(
    HTML(
        '<b>Review <a target="blank" href="https://console.aws.amazon.com/sagemaker/home?region={}#/processing-jobs/{}">Processing Job</a></b>'.format(
            region, scikit_processing_job_name
        )
    )
)

# In[15]:


from IPython.core.display import display, HTML

display(
    HTML(
        '<b>Review <a target="blank" href="https://console.aws.amazon.com/cloudwatch/home?region={}#logStream:group=/aws/sagemaker/ProcessingJobs;prefix={};streamFilter=typeLogStreamPrefix">CloudWatch Logs</a> After About 5 Minutes</b>'.format(
            region, scikit_processing_job_name
        )
    )
)

# In[16]:


from IPython.core.display import display, HTML

display(
    HTML(
        '<b>Review <a target="blank" href="https://s3.console.aws.amazon.com/s3/buckets/{}/{}/?region={}&tab=overview">S3 Output Data</a> After The Processing Job Has Completed</b>'.format(
            bucket, scikit_processing_job_name, region
        )
    )
)

# # Monitor the Processing Job

# In[17]:


running_processor = sagemaker.processing.ProcessingJob.from_processing_name(
    processing_job_name=scikit_processing_job_name, sagemaker_session=sess
)

processing_job_description = running_processor.describe()

print(processing_job_description)

# In[18]:


running_processor.wait(logs=False)

# # _Please Wait Until the ^^ Processing Job ^^ Completes Above._

# # Inspect the Processed Output Data
# 
# Take a look at a few rows of the transformed dataset to make sure the processing was successful.

# In[19]:


processing_job_description = running_processor.describe()

output_config = processing_job_description["ProcessingOutputConfig"]
for output in output_config["Outputs"]:
    if output["OutputName"] == "train":
        processed_train_data_s3_uri = output["S3Output"]["S3Uri"]
    if output["OutputName"] == "validation":
        processed_validation_data_s3_uri = output["S3Output"]["S3Uri"]
    if output["OutputName"] == "test":
        processed_test_data_s3_uri = output["S3Output"]["S3Uri"]

print(processed_train_data_s3_uri)
print(processed_validation_data_s3_uri)
print(processed_test_data_s3_uri)

# In[20]:


!aws s3 ls $processed_train_data_s3_uri/

# In[21]:


!aws s3 ls $processed_validation_data_s3_uri/

# In[22]:


!aws s3 ls $processed_test_data_s3_uri/

# # Pass Variables to the Next Notebook(s)

# In[23]:


%store raw_input_data_s3_uri

# In[24]:


%store train_split_percentage

# In[25]:


%store validation_split_percentage

# In[26]:


%store test_split_percentage

# In[27]:


# %store balance_dataset

# In[28]:


%store processed_train_data_s3_uri

# In[29]:


%store processed_validation_data_s3_uri

# In[30]:


%store processed_test_data_s3_uri

# In[31]:


%store

# In[ ]:



