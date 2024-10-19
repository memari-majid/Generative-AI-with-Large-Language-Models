#!/usr/bin/env python
# coding: utf-8

# # Train Falcon with near-linear scaling using Sharded Data Parallelism technique in SageMaker Model Parallelism Library

# In this notebook, you'll learn how to train the Hugging Face Transformers [Falcon](https://huggingface.co/docs/transformers/main/model_doc/falcon) model with the [Sharded Data Parallelism](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-sharded-data-parallelism.html) technique supported by [SageMaker's Model Parallelism library (SMP)](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html) with PyTorch 2.0 and [GLUE/SST2 dataset](https://huggingface.co/datasets/glue/viewer/sst2/train) on SageMaker. 
# 
# Sharded data parallelism is a distributed training technique that splits the model parameters, gradients, and optimizer states across GPUs in a data parallel group. It is purpose-built for extreme-scale models and leverages Amazon in-house [MiCS](https://arxiv.org/pdf/2205.00119.pdf) technology which achieves a near-linear scaling efficiency. For large models that cannot fit into a single GPU, we also recommend using the sharded data parallelism technique with [Activation Checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-activation-checkpointing.html) and [Activation Offloading](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-activation-offloading.html) in SMP first, before leveraging other techniques such as tensor parallelism or pipeline parallelism.
# 
# This feature is also compatible with [Tensor Parallelism](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-tensor-parallelism.html). 
# 
# This notebook is accompanied by the following files:
# 
# - `train.py`: The entry point script that'll be passed to the SageMaker PyTorch estimator later in this notebook when launching the training job. This script is prepared to run an end-to-end training of the Falcon model with SMP, settings for sharded data parallelism applied, and implemented with code lines to save, load, and fine-tune the model. You can follow the comments throughout the script to learn where the SMP APIs and code modifications are implemented.
# - `data_pipeline.py`: This has data pipeline functions to prepare the training dataset.
# - `learining_rate.py`: This has functions for learning rate schedule.
# - `requirements.txt`: This installs the dependencies, including huggingface transformers.
# - `memory_tracker.py`: This has functions to track memory usage.
# - `model_config.py`: This has functions to get model configuration information.
# - `sdp_utils.py`: This has util functions for sharded data parallelism.
# 
# ### Additional resources
# - To learn more about the SageMaker model parallelism library, see [Model Parallel Distributed Training with SageMaker Distributed](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html).
# 
# - To learn more about using the SageMaker Python SDK with PyTorch, see [Using PyTorch with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html).
# 
# - To learn more about launching a training job in Amazon SageMaker with your own training image, see [Use Your Own Training Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html).
# 
# - To learn more about sharded data parallelism, check [Sharded Data Parallelism](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-sharded-data-parallelism.html) or the blog [Near-linear scaling of gigantic-model training on AWS](https://www.amazon.science/blog/near-linear-scaling-of-gigantic-model-training-on-aws).
# 
# ### Prerequisites
# You must create an S3 bucket to store the input data for training. This bucket must be located in the same AWS Region that you choose to launch your training job. To learn how to create a S3 bucket, see [Create your first S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html) in the *Amazon S3 documentation*.
# 

# ## Amazon SageMaker initialization
# 
# Run the following cell to import SageMaker modules and retrieve information of your current SageMaker work environment, such as your AWS account ID, the AWS Region, and the ARN of your Amazon SageMaker execution role. Upgrade SageMaker SDK to the latest version. 
# 
# **NOTE:** This step might require a kernel restart.

# In[2]:


# %pip install --upgrade sagemaker
# %pip install sagemaker-experiments

# In[3]:


%%time
import os

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

role = (
    get_execution_role()
)  # provide a pre-existing role ARN as an alternative to creating a new role
print(f"SageMaker Execution Role: {role}")

client = boto3.client("sts")
account = client.get_caller_identity()["Account"]
print(f"AWS account: {account}")

session = boto3.session.Session()
region = session.region_name
print(f"AWS region: {region}")

sm_boto_client = boto3.client("sagemaker")
sagemaker_session = sagemaker.session.Session(boto_session=session)

# get default bucket
default_bucket = sagemaker_session.default_bucket()
print()
print("Default bucket for this session: ", default_bucket)

# ## Download and prepare GLUE/SST2 data
# Here you will download, prepare the GLUE/SST2 dataset and then copy the files to S3.

# ### Install the Hugging Face Transformers and Datasets libraries

# In[4]:


! pip install -q datasets transformers==4.21.0

# In[5]:


import datasets
from datasets import load_dataset, load_from_disk, load_metric

# In[6]:


from sagemaker.pytorch import PyTorch
import transformers
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from transformers.testing_utils import CaptureLogger

# In[7]:


logger = logging.getLogger(__name__)

# ### Load data
# This section loads the [GLUE/SST2](https://huggingface.co/datasets/glue/viewer/sst2/train) dataset and splits it to training and validation datasets.

# In[8]:


hyperparameters = {
    "dataset_name": "glue",
    "dataset_config_name": "sst2",
    "do_train": True,
    "do_eval": True,
    "cache_dir": "tmp",
}

# In[9]:


raw_datasets = load_dataset(
    hyperparameters["dataset_name"],
    hyperparameters["dataset_config_name"],
)

# In[10]:


if "validation" not in raw_datasets.keys():
    raw_datasets["validation"] = load_dataset(
        hyperparameters["dataset_name"],
        hyperparameters["dataset_config_name"],
        split="train[:5%]",
        cache_dir=hyperparameters["cache_dir"],
    )

    raw_datasets["train"] = load_dataset(
        hyperparameters["dataset_name"],
        hyperparameters["dataset_config_name"],
        split="train[5%:]",
        cache_dir=hyperparameters["cache_dir"],
    )

# ### Load tokenizer
# Nearly every NLP task begins with a tokenizer. A tokenizer converts your text data into a format (token) that can be processed by the NLP model.
# The following cell loads a tokenizer for Falcon using [AutoTokenizer.from_pretrained()](https://huggingface.co/docs/transformers/v4.19.4/en/autoclass_tutorial#autotokenizer).

# In[11]:


tokenizer_kwargs = {
    "cache_dir": hyperparameters["cache_dir"],
}

tokenizer = AutoTokenizer.from_pretrained(
    "tiiuae/falcon-40b", trust_remote_code=True, **tokenizer_kwargs
)

# ### Preprocess data
# 
# The following two cells set up a function to run the tokenizer and group texts into chunks smaller than the block size.

# In[12]:


def tokenize_function(examples):
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
    return output


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
    result["labels"] = result["input_ids"].copy()
    return result

# In[13]:


column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

# since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    remove_columns=column_names,
    desc="Running tokenizer on dataset",
)


block_size = tokenizer.model_max_length
if block_size > 1024:
    logger.warning(
        f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
        "Picking 1024 instead. You can change that default value by passing --block_size xxx."
    )
    block_size = 1024
else:
    if block_size > tokenizer.model_max_length:
        logger.warning(
            f"The block_size passed ({block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(block_size, tokenizer.model_max_length)

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    #     num_proc=args.preprocessing_num_workers,
    desc=f"Grouping texts in chunks of {block_size}",
)

# Set additional hyperparameters and S3 paths for mapping the train and validation datasets properly depending on the phase (training or validation) of the training job in each epoch.

# In[14]:


if hyperparameters["do_train"]:
    if "train" not in tokenized_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = lm_datasets["train"]


if hyperparameters["do_eval"]:
    if "validation" not in tokenized_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = lm_datasets["validation"]

# In[15]:


training_dataset_location = None
validation_dataset_location = None


if hyperparameters["do_train"]:
    train_dataset.to_json("./training.json")
    training_dataset_location = "s3://{}/dataset/train/".format(default_bucket)

if hyperparameters["do_eval"]:
    eval_dataset.to_json("./validation.json")
    validation_dataset_location = "s3://{}/dataset/validation/".format(default_bucket)

# In[16]:


if training_dataset_location is not None:
    command = "aws s3 cp ./training.json {}".format(training_dataset_location)
    os.system(command)

if validation_dataset_location is not None:
    command = "aws s3 cp ./validation.json {}".format(validation_dataset_location)
    os.system(command)

# In[17]:


if hyperparameters["do_train"]:
    command = "rm ./training.json"
    os.system(command)

if hyperparameters["do_eval"]:
    command = "rm ./validation.json"
    os.system(command)

# In[18]:


%store training_dataset_location
%store validation_dataset_location

# In[19]:


%store

# ## Specify Amazon S3 bucket paths

# Here you need to specify the paths for training data to be used by your job. The bucket used must be in the same region as where training will run. In the cells above you downloaded the GLUE/SST2 training and validation split datasets and uploaded the json files in an S3 bucket in your account. This example will train on those json files.
# 
# After you successfully run this example tensor parallel training job, you can modify the S3 bucket to where your own dataset is stored.

# In[20]:


%store -r training_dataset_location
%store -r validation_dataset_location

# if you're bringing your own data, uncomment the following lines and specify the locations there
# training_dataset_location = YOUR_S3_BUCKET/training
# validation_dataset_location = YOUR_S3_BUCKET/validation

# In[21]:


s3_train_bucket = training_dataset_location
s3_test_bucket = validation_dataset_location

# The following S3 bucket will store the output artifacts of the training job. You can modify this as needed.

# In[22]:


s3_output_bucket = f"s3://sagemaker-{region}-{account}/smp-tensorparallel-outputdir/"

# ## Define data channels for SageMaker Training using Amazon S3
# 
# In this step, define SageMaker training data channels to the S3 buckets.  

# In[23]:


# Set use_fsx to False by default
# Set below var to True if you want to use fsx (see next cell)
use_fsx = False
if not use_fsx:
    if s3_train_bucket != None:
        train = sagemaker.inputs.TrainingInput(
            s3_train_bucket, distribution="FullyReplicated", s3_data_type="S3Prefix"
        )
        data_channels = {"train": train}
    else:
        data_channels = {"train": mock_data}
    if s3_test_bucket != None:
        test = sagemaker.inputs.TrainingInput(
            s3_test_bucket, distribution="FullyReplicated", s3_data_type="S3Prefix"
        )
        data_channels["test"] = test
    else:
        data_channels["test"] = mock_data
    print(data_channels)

# ## (Optional) Set up and use Amazon FSx for data channels and checkpoints
# 
# While the previous option of using Amazon S3 is easier to set up, using an FSx can be beneficial for performance when dealing with large input sizes and large model sizes. If you are using models above 13B, checkpointing should be done using FSx. 
# 
# Please see the instructions from [Distributed Training of Mask-RCNN in Amazon SageMaker Using FSx](https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/distributed_tensorflow_mask_rcnn/mask-rcnn-scriptmode-fsx.ipynb) to create an FSx Lustre file system and import the dataset from the S3 bucket to your FSx file system. Note that the FSx file system must be created in a private subnet with internet gateway to ensure that training job has access to the internet. For general guidance on setting an FSx Lustre file system as data input channel, see [Configure Data Input Channel to Use Amazon FSx for Lustre](https://docs.aws.amazon.com/sagemaker/latest/dg/model-access-training-data.html#model-access-training-data-fsx).

# In[24]:


# Instructions obtained from:
# https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/distributed_tensorflow_mask_rcnn/mask-rcnn-scriptmode-fsx.ipynb

if use_fsx:
    from sagemaker.inputs import FileSystemInput

    # Specify FSx Lustre file system id.
    file_system_id = "<your-file-system-id>"

    # Specify the SG and subnet used by the FSX, these are passed to SM Estimator so jobs use this as well
    fsx_security_group_id = "<your-security-group-id>"
    fsx_subnet = "<your-subnet>"

    # Specify directory path for input data on the file system.
    # You need to provide normalized and absolute path below.
    # Your mount name can be provided by you when creating fsx, or generated automatically.
    # You can find this mount_name on the FSX page in console.
    # Example of fsx generated mount_name: "3x5lhbmv"
    base_path = "<your-mount-name>"

    # Specify your file system type.
    file_system_type = "FSxLustre"

    train = FileSystemInput(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=base_path,
        file_system_access_mode="rw",
    )

    data_channels = {"train": train, "test": train}

# ## Set hyperparameters, metric definitions, and MPI options
# The following `hyperparameters` dictionary passes arguments to the training script (`train.py`) and set the model parallel configuration when creating the training job.
# 
# You can also add custom `mpi` flags. By default, we have `--mca btl_vader_single_copy_mechanism none` to remove unnecessary logs.
# 
# Next, we add a base metric definitions to enable the metric upload in SageMaker. You can add any further metric definitions.
# 
# Note that we add the `sharded_data_parallel_degree` parameter to the `hyperparameter` dictionary. This will be parsed and used when we configure a SageMaker PyTorch estimator to activate sharded data parallelism.

# In[25]:


hyperparameters = {
    "max_steps": 100,
    "seed": 12345,
    "fp16": 0,
    "bf16": 1,
    "lr": 2.0e-4,
    "lr_decay_iters": 125000,
    "min_lr": 0.00001,
    "lr-decay-style": "linear",
    "warmup": 0.01,
    "num_kept_checkpoints": 5,
    "checkpoint_freq": 200,
    "logging_freq": 1,
    "save_final_full_model": 0,
    "delayed_param": 1,
    "offload_activations": 0,
    "activation_loading_horizon": 4,
    "gradient_accumulation": 1,
    "validation_freq": 200,
    "train_batch_size": 4,
    "val_batch_size": 4,
    "zipped_data": 0,
    "epochs": 100,
    "use_distributed_transformer": 0,
    "model_type": "falcon",
    # parameters for sharded data parallelism
    "sharded_data_parallel_degree": 16,
}

if use_fsx:
    # make sure to update paths for training-dir and test-dir based on the paths of datasets in fsx
    # If you want to resume training, set checkpoint-dir to the same path as a previous job.
    SM_TRAIN_DIR = "/opt/ml/input/data/train"
    hyperparameters["checkpoint-dir"] = f"{SM_TRAIN_DIR}/checkpointdir-job2"
    hyperparameters["model-dir"] = f"{SM_TRAIN_DIR}/modeldir-job2"
    hyperparameters["training-dir"] = f"{SM_TRAIN_DIR}/datasets/pytorch_gpt/train_synthetic"
    hyperparameters["test-dir"] = f"{SM_TRAIN_DIR}/datasets/pytorch_gpt/val_synthetic"

# The checkpoint path (hyperparameters['checkpoint-dir'] or checkpoint_s3_uri) is not unique per job.
# You need to modify as needed for different runs.
# If same path is used for unrelated runs, this may increase time when downloading unnecessary checkpoints,
# and cause conflicts when loading checkpoints.

mpioptions = "-x NCCL_DEBUG=WARN -x SMDEBUG_LOG_LEVEL=ERROR "
mpioptions += (
    "-x SMP_DISABLE_D2D=1 -x SMP_D2D_GPU_BUFFER_SIZE_BYTES=1 -x SMP_NCCL_THROTTLE_LIMIT=1 "
)
mpioptions += "-x FI_EFA_USE_DEVICE_RDMA=1 -x FI_PROVIDER=efa -x RDMAV_FORK_SAFE=1"

metric_definitions = [
    {"Name": "base_metric", "Regex": "<><><><><><>"}
]  # Add your custom metric definitions

# Set the model configuration.

# In[26]:


model_config = "falcon-7b"

if model_config == "falcon-7b":
    model_params = {
        "max_context_width": 2048,
        "hidden_width": 4544,
        "num_layers": 32,
        "num_heads": 71,
        "num_heads_kv": 71,
    }
else:
    raise RuntimeError("Unknown model config")

for k, v in model_params.items():
    hyperparameters[k] = v

# ## Specify essential parameters for a SageMaker Training job
# 
# Next, you use the [`SageMaker Estimator class`](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html) to define a SageMaker Training Job, passing values through the following parameters for training job name, the number of EC2 instances, the instance type, and the size of the volume attached to the instances. 
# 
# * `instance_count`
# * `instance_type`
# * `volume_size`
# * `base_job_name`
# 
# ### Update the type and the number of EC2 instance to use
# 
# The instance type and the number of instances you specify to the `instance_type` and `instance_count` parameters, respectively, determine the total number of GPUs (world size).
# 
# $$ \text{(world size) = (the number of GPUs on a single instance)}\times\text{(the number of instances)}$$

# In[27]:


instance_type = "ml.p4d.24xlarge"

instance_count = 2

# set to the number of GPUs on that instance
processes_per_host = 8

# To look up the number of GPUs of different instance types, see [Amazon EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/). Use the section **Accelerated Computing** to see general purpose GPU instances. Note that, for example, a given instance type `p4d.24xlarge` has a corresponding instance type `ml.p4d.24xlarge` in SageMaker.
# For SageMaker supported `ml` instances and cost information, see [Amazon SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/). 

# ### Specify a base job name

# In[28]:


machine_str = instance_type.split(".")[1] + instance_type.split(".")[2][:3]
sharding_degree = hyperparameters["sharded_data_parallel_degree"]
base_job_name = (
    f'smp-{model_config}-{machine_str}-sdp{sharding_degree}-bs{hyperparameters["train_batch_size"]}'
)

# In[29]:


if not use_fsx:
    # If you want to resume training, set checkpoint_s3_uri to the same path as a previous job.
    # Previous checkpoint to load must have same model config.
    checkpoint_bucket = f"s3://sagemaker-{region}-{account}/"
    checkpoint_s3_uri = (
        f"{checkpoint_bucket}/experiments/gpt_synthetic_simpletrainer_checkpoints/{base_job_name}/"
    )

# In[30]:


print(f"base_job_name: {base_job_name} checkpoint_s3_uri: {checkpoint_s3_uri}")

# ### Create a SageMaker PyTorch estimator
# 
# The following cell constructs a PyTorch estimator using the parameters defined above. To see how the SageMaker APIs and functions are applied to the script, see the `train.py` file.

# In[31]:


kwargs = {}
if use_fsx:
    # Use the security group and subnet that was used to create the fsx filesystem
    kwargs["security_group_ids"] = [fsx_security_group_id]
    kwargs["subnets"] = [fsx_subnet]

smp_estimator = PyTorch(
    entry_point="train.py",
    source_dir=os.getcwd(),
    role=role,
    instance_type=instance_type,
    instance_count=instance_count,
    sagemaker_session=sagemaker_session,
    distribution={
        "mpi": {
            "enabled": True,
            "processes_per_host": processes_per_host,
            "custom_mpi_options": mpioptions,
        },
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": {
                    "ddp": True,
                    "skip_tracing": True,
                    "delayed_parameter_initialization": hyperparameters["delayed_param"] > 0,
                    "offload_activations": hyperparameters["offload_activations"] > 0,
                    "activation_loading_horizon": hyperparameters["activation_loading_horizon"],
                    "sharded_data_parallel_degree": hyperparameters["sharded_data_parallel_degree"],
                    "fp16": hyperparameters["fp16"] > 0,
                    "bf16": hyperparameters["bf16"] > 0,
                    # partitions is a required param in the current SM SDK so it needs to be passed,
                    "partitions": 1,
                },
            }
        },
    },
    framework_version="2.0",
    py_version="py310",
    output_path=s3_output_bucket,
    checkpoint_s3_uri=checkpoint_s3_uri if not use_fsx else None,
    checkpoint_local_path=hyperparameters["checkpoint-dir"] if use_fsx else None,
    metric_definitions=metric_definitions,
    hyperparameters=hyperparameters,
    debugger_hook_config=False,
    disable_profiler=True,
    base_job_name=base_job_name,
    **kwargs,
)

# Finally, run the `estimator.fit` method to launch the SageMaker training job of the Falcon model with sharded data parallelism.

# In[32]:


smp_estimator.fit(inputs=data_channels, logs=True)

# ## Accessing the Training Logs
# 
# You can access the training logs from [Amazon CloudWatch](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/WhatIsCloudWatch.html). Make sure to look at the logs of **algo-1** because that is the main node whose output stream has the training job logs.
# 
# You can use CloudWatch to track SageMaker GPU and memory utilization during training and inference. To view the metrics and logs that SageMaker writes to CloudWatch, see [SageMaker Jobs and Endpoint Metrics](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html#cloudwatch-metrics-jobs) in the Amazon SageMaker Developer Guide.
# 
# If you are a new user of CloudWatch, see [Getting Started with Amazon CloudWatch](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/GettingStarted.html). 
# 
# For additional information on monitoring and analyzing Amazon SageMaker training jobs, see [Monitor and Analyze Training Jobs Using Metrics](https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html).
# 
# ## Deploying Trained Model for Inference
# 
# In most cases, a trained model can be deployed on a single device for inference because inference only requires a small amount of memory. You can use the SMP API to create a single, unified model after training: the [smp.DistributedModel.save_model()](https://sagemaker.readthedocs.io/en/stable/api/training/smp_versions/latest/smd_model_parallel_tensorflow.html#smp.DistributedModel.save_model) method for TensorFlow, and the [smp.save()](https://sagemaker.readthedocs.io/en/stable/api/training/smp_versions/latest/smd_model_parallel_pytorch.html#apis-for-saving-and-loading) function for PyTorch.
# 
# After you build and train your models, you can deploy them to get predictions in one of two ways:
# 
# * To set up a persistent endpoint to get predictions from your models, use SageMaker hosting services. For an overview on deploying a single model or multiple models with SageMaker hosting services, see [Deploy a Model on SageMaker Hosting Services](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-deployment.html#how-it-works-hosting).
# * To get predictions for an entire dataset, use SageMaker batch transform. For an overview on deploying a model with SageMaker Batch Transform, see [Get Inferences for an Entire Dataset with Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html).
# 
# To learn more about deploying models for inference using SageMaker, see [Deploy Models for Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html). 
# 

# In[ ]:



