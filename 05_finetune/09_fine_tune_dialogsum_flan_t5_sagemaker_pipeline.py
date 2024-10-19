#!/usr/bin/env python
# coding: utf-8

# # Creating an Automated Model Fine-Tuning Workflow with SageMaker Pipelines

# # NOTE:  THIS NOTEBOOK WILL TAKE ABOUT 30 MINUTES TO COMPLETE.
# 
# # PLEASE BE PATIENT.

# # SageMaker Pipelines
# 
# Amazon SageMaker Pipelines support the following:
# 
# * **Pipelines** - A Directed Acyclic Graph of steps and conditions to orchestrate SageMaker jobs and resource creation.
# * **Processing Job Steps** - A simplified, managed experience on SageMaker to run data processing workloads, such as feature engineering, data validation, model evaluation, and model interpretation.
# * **Training Job Steps** - An iterative process that teaches a model to make predictions by presenting examples from a training dataset.
# * **Conditional Steps** - Provides conditional execution of branches in a pipeline.
# * **Registering Models** - Creates a model package resource in the Model Registry that can be used to create deployable models in Amazon SageMaker.
# * **Parameterized Executions** - Allows pipeline executions to vary by supplied parameters.
# * **Transform Job Steps** - A batch transform to preprocess datasets to remove noise or bias that interferes with training or inference from your dataset, get inferences from large datasets, and run inference when you don't need a persistent endpoint.
# 
# # Our Pipeline
# 
# In the Processing Step, we perform Feature Engineering to tokenizer our dialogue inputs using the `transformer` library from HuggingFace/
# 
# In the Training Step, we fine-tune the model to summarize dialogue effectively on the `diagsum` dataset.
# 
# In the Evaluation Step, we take the fine-tuned model and a test dataset as input, and produce a JSON file containing evaluation metrics based on the ROUGE metric for summarization.
# 
# In the Condition Step, we decide whether to register this model if the metrics of the model, as determined by our evaluation step, exceeded some value. 

# In[14]:


# %pip install sagemaker-experiments==0.1.45

# In[15]:


from botocore.exceptions import ClientError

import os
import sagemaker
import logging
import boto3
import sagemaker
import pandas as pd

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
s3 = boto3.Session().client(service_name="s3", 
                            region_name=region,
                            config=config)

# # Set S3 Source Location

# In[16]:


%store -r raw_input_data_s3_uri

# In[17]:


try:
    raw_input_data_s3_uri
except NameError:
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    print("[ERROR] YOU HAVE TO RUN THE PREVIOUS NOTEBOOK ")
    print("You did not have the required datasets.       ")
    print("++++++++++++++++++++++++++++++++++++++++++++++")

# In[18]:


print(raw_input_data_s3_uri)

# In[19]:


if not raw_input_data_s3_uri:
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    print("[ERROR] YOU HAVE TO RUN THE PREVIOUS NOTEBOOK ")
    print("You did not have the required datasets.       ")
    print("++++++++++++++++++++++++++++++++++++++++++++++")
else:
    print("[OK]")

# # Track the Pipeline as an `Experiment`

# In[20]:


import time

# In[21]:


running_executions = 0
completed_executions = 0

try:
    existing_pipeline_executions_response = sm.list_pipeline_executions(
        PipelineName=pipeline_name,
        SortOrder="Descending",
    )

    if "PipelineExecutionSummaries" in existing_pipeline_executions_response.keys():
        if len(existing_pipeline_executions_response["PipelineExecutionSummaries"]) > 0:
            execution = existing_pipeline_executions_response["PipelineExecutionSummaries"][0]
            if "PipelineExecutionStatus" in execution:
                if execution["PipelineExecutionStatus"] == "Executing":
                    running_executions = running_executions + 1
                else:
                    completed_executions = completed_executions + 1

            print(
                "[INFO] You have {} Pipeline execution(s) currently running and {} execution(s) completed.".format(
                    running_executions, completed_executions
                )
            )
    else:
        print("[OK] Please continue.")
except:
    pass

if running_executions == 0:
    timestamp = int(time.time())
    pipeline_name = "dialogue-summary-pipeline-{}".format(timestamp)
    print("Created Pipeline Name: " + pipeline_name)

# In[22]:


print(pipeline_name)

# In[23]:


%store pipeline_name

# In[24]:


from smexperiments.experiment import Experiment

pipeline_experiment = Experiment.create(
    experiment_name=pipeline_name,
    description="Dialogue Summarization Pipeline Experiment",
    sagemaker_boto_client=sm,
)
pipeline_experiment_name = pipeline_experiment.experiment_name
print("Created Pipeline Experiment Name: {}".format(pipeline_experiment_name))

# In[25]:


print(pipeline_experiment_name)

# In[26]:


%store pipeline_experiment_name

# # Create the `Trial`

# In[27]:


from smexperiments.trial import Trial

# In[28]:


%store -r pipeline_trial_name

timestamp = int(time.time())
pipeline_trial = Trial.create(
    trial_name="trial-{}".format(timestamp), experiment_name=pipeline_experiment_name, sagemaker_boto_client=sm
)
pipeline_trial_name = pipeline_trial.trial_name
print("Created Trial Name: {}".format(pipeline_trial_name))

# In[29]:


print(pipeline_trial_name)

# In[30]:


%store pipeline_trial_name

# # Define Parameters to Parametrize Pipeline Execution
# 
# We define Workflow Parameters by which we can parametrize our Pipeline and vary the values injected and used in Pipeline executions and schedules without having to modify the Pipeline definition.
# 
# The supported parameter types include:
# 
# * `ParameterString` - representing a `str` Python type
# * `ParameterInteger` - representing an `int` Python type
# * `ParameterFloat` - representing a `float` Python type
# 
# These parameters support providing a default value, which can be overridden on pipeline execution. The default value specified should be an instance of the type of the parameter.

# In[31]:


from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat
)

# # Feature Engineering Step

# In[32]:


%store -r raw_input_data_s3_uri

# In[33]:


print(raw_input_data_s3_uri)

# In[34]:


!aws s3 ls $raw_input_data_s3_uri

# # Setup Pipeline Parameters
# These parameters are used by the entire pipeline.

# In[35]:


model_checkpoint='google/flan-t5-base'

# In[36]:


model_checkpoint = ParameterString(
    name="ModelCheckpoint",
    default_value=model_checkpoint,
)

# # Setup Processing Parameters

# In[37]:


input_data = ParameterString(
    name="InputData",
    default_value=raw_input_data_s3_uri,
)

processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount",
    default_value=1,
)

processing_instance_type = ParameterString(
    name="ProcessingInstanceType",
    default_value="ml.c5.2xlarge",
)

train_split_percentage = ParameterFloat(
    name="TrainSplitPercentage",
    default_value=0.90,
)

validation_split_percentage = ParameterFloat(
    name="ValidationSplitPercentage",
    default_value=0.05,
)

test_split_percentage = ParameterFloat(
    name="TestSplitPercentage",
    default_value=0.05,
)

# We create an instance of an `SKLearnProcessor` processor and we use that in our `ProcessingStep`.

# In[38]:


from sagemaker.sklearn.processing import SKLearnProcessor

processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    env={"AWS_DEFAULT_REGION": region},
    max_runtime_in_seconds=432000,
)

# ### _Ignore any `WARNING` ^^ above ^^._

# ### Setup Pipeline Step Caching
# Cache pipeline steps for a duration of time using [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#Durations) format.  
# 
# More details on SageMaker Pipeline step caching here:  https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-caching.html

# In[39]:


from sagemaker.workflow.steps import CacheConfig

cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")

# Finally, we use the processor instance to construct a `ProcessingStep`, along with the input and output channels and the code that will be executed when the pipeline invokes pipeline execution. This is very similar to a processor instance's `run` method, for those familiar with the existing Python SDK.
# 
# Note the `input_data` parameters passed into `ProcessingStep` as the input data of the step itself. This input data will be used by the processor instance when it is run.
# 
# Also, take note the `"train"`, `"validation"` and `"test"` named channels specified in the output configuration for the processing job. Such step `Properties` can be used in subsequent steps and will resolve to their runtime values at execution. In particular, we'll call out this usage when we define our training step.

# In[40]:


from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

processing_inputs = [
    ProcessingInput(
        input_name="raw-input-data",
        source=input_data,
        destination="/opt/ml/processing/input/data/",
        s3_data_distribution_type="ShardedByS3Key",
    )
]

processing_outputs = [
    ProcessingOutput(
        output_name="train",
        s3_upload_mode="EndOfJob",
        source="/opt/ml/processing/output/data/train",
    ),
    ProcessingOutput(
        output_name="validation",
        s3_upload_mode="EndOfJob",
        source="/opt/ml/processing/output/data/validation",
    ),
    ProcessingOutput(
        output_name="test",
        s3_upload_mode="EndOfJob",
        source="/opt/ml/processing/output/data/test",
    ),
]

processing_step = ProcessingStep(
    name="Processing",
    code="preprocess.py",
    processor=processor,
    inputs=processing_inputs,
    outputs=processing_outputs,
    job_arguments=[
        "--train-split-percentage",
        str(train_split_percentage.default_value),
        "--validation-split-percentage",
        str(validation_split_percentage.default_value),
        "--test-split-percentage",
        str(test_split_percentage.default_value),
        "--model-checkpoint",
        str(model_checkpoint.default_value),
    ],
    cache_config=cache_config
)

print(processing_step)

# # Train Step

# # Setup Training Hyper-Parameters

# In[41]:


train_instance_type = ParameterString(name="TrainInstanceType", default_value="ml.c5.9xlarge")
train_instance_count = ParameterInteger(name="TrainInstanceCount", default_value=1)

# In[42]:


epochs = ParameterInteger(name="Epochs", default_value=1)
learning_rate = ParameterFloat(name="LearningRate", default_value=0.00001)
weight_decay = ParameterFloat(name="WeightDecay", default_value=0.01)
train_batch_size = ParameterInteger(name="TrainBatchSize", default_value=4)
validation_batch_size = ParameterInteger(name="ValidationBatchSize", default_value=4)
test_batch_size = ParameterInteger(name="TestBatchSize", default_value=4)
train_volume_size = ParameterInteger(name="TrainVolumeSize", default_value=1024)
input_mode = ParameterString(name="InputMode", default_value="FastFile")
train_sample_percentage = ParameterFloat(name="TrainSamplePercentage", default_value=0.01)

# ### Setup Metrics To Track Model Performance

# In[43]:


metrics_definitions = [
    {"Name": "train:loss", "Regex": "'train_loss': ([0-9\\.]+)"},
    {"Name": "validation:loss", "Regex": "'eval_loss': ([0-9\\.]+)"},
]

# ### Create the Estimator
# 
# We configure an Estimator and the input dataset. A typical training script loads data from the input channels, configures training with hyperparameters, trains a model, and saves a model to `model_dir` so that it can be hosted later.
# 
# We also specify the model path where the models from training will be saved.
# 
# Note the `train_instance_type` parameter passed may be also used and passed into other places in the pipeline. In this case, the `train_instance_type` is passed into the estimator.

# In[44]:


from sagemaker.pytorch import PyTorch
import uuid

checkpoint_s3_prefix = "checkpoints/{}".format(str(uuid.uuid4()))
checkpoint_s3_uri = "s3://{}/{}/".format(bucket, checkpoint_s3_prefix)

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    role=role,
    instance_count=train_instance_count,
    instance_type=train_instance_type,
    volume_size=train_volume_size,
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

# ### Configure Training Step
# 
# Finally, we use the estimator instance to construct a `TrainingStep` as well as the `Properties` of the prior `ProcessingStep` used as input in the `TrainingStep` inputs and the code that will be executed when the pipeline invokes pipeline execution. This is very similar to an estimator's `fit` method, for those familiar with the existing Python SDK.
# 
# In particular, we pass in the `S3Uri` of the `"train"`, `"validation"` and `"test"` output channel to the `TrainingStep`. The `properties` attribute of a Workflow step match the object model of the corresponding response of a describe call. These properties can be referenced as placeholder values and are resolved, or filled in, at runtime. For example, the `ProcessingStep` `properties` attribute matches the object model of the [DescribeProcessingJob](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeProcessingJob.html) response object.

# In[45]:


from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep

training_step = TrainingStep(
    name="Train",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        ),
        "validation": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
        ),
        "test": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
        ),
    },
    cache_config=cache_config,
)

print(training_step)

# # Evaluation Step
# 
# First, we develop an evaluation script that will be specified in a Processing step that will perform the model evaluation.
# 
# The evaluation script `evaluate_model_metrics.py` takes the trained model and the test dataset as input, and produces a JSON file containing evaluation metrics.
# 
# After pipeline execution, we will examine the resulting `evaluation.json` for analysis.
# 
# The evaluation script:
# 
# * loads in the model
# * reads in the test data
# * issues a bunch of predictions against the test data
# * builds an evaluation report
# * saves the evaluation report to the evaluation directory
# 
# Next, we create an instance of a `SKLearnProcessor` and we use that in our `ProcessingStep`.
# 
# Note the `processing_instance_type` parameter passed into the processor.

# In[46]:


from sagemaker.sklearn.processing import SKLearnProcessor

evaluation_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    env={"AWS_DEFAULT_REGION": region},
    max_runtime_in_seconds=432000,
)

# ### _Ignore any `WARNING` ^^ above ^^._

# We use the processor instance to construct a `ProcessingStep`, along with the input and output channels and the code that will be executed when the pipeline invokes pipeline execution. This is very similar to a processor instance's `run` method, for those familiar with the existing Python SDK.
# 
# The `TrainingStep` and `ProcessingStep` `properties` attribute matches the object model of the [DescribeTrainingJob](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTrainingJob.html) and  [DescribeProcessingJob](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeProcessingJob.html) response objects, respectively.

# In[47]:


from sagemaker.workflow.properties import PropertyFile

evaluation_report = PropertyFile(name="EvaluationReport", output_name="metrics", path="evaluation.json")

# In[48]:


evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=evaluation_processor,
    code="evaluate_model_metrics.py",
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/input/model"            
            
        ),
        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/input/data"       
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/metrics/",
            output_name="metrics", 
            s3_upload_mode="EndOfJob"            
        ),
    ],
    property_files=[evaluation_report],
)

# In[49]:


from sagemaker.model_metrics import MetricsSource, ModelMetrics

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json",
    )
)

print(model_metrics)

# # Register Model Step
# 
# ![](img/pipeline-5.png)
# 
# We use the estimator instance that was used for the training step to construct an instance of `RegisterModel`. The result of executing `RegisterModel` in a pipeline is a Model Package. A Model Package is a reusable model artifacts abstraction that packages all ingredients necessary for inference. Primarily, it consists of an inference specification that defines the inference image to use along with an optional model weights location.
# 
# A Model Package Group is a collection of Model Packages. You can create a Model Package Group for a specific ML business problem, and you can keep adding versions/model packages into it. Typically, we expect customers to create a ModelPackageGroup for a SageMaker Workflow Pipeline so that they can keep adding versions/model packages to the group for every Workflow Pipeline run.
# 
# The construction of `RegisterModel` is very similar to an estimator instance's `register` method, for those familiar with the existing Python SDK.
# 
# In particular, we pass in the `S3ModelArtifacts` from the `TrainingStep`, `step_train` properties. The `TrainingStep` `properties` attribute matches the object model of the [DescribeTrainingJob](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTrainingJob.html) response object.
# 
# Of note, we provided a specific model package group name which we will use in the Model Registry and CI/CD work later on.

# In[50]:


model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

deploy_instance_type = ParameterString(name="DeployInstanceType", default_value="ml.m5.4xlarge")

deploy_instance_count = ParameterInteger(name="DeployInstanceCount", default_value=1)

# In[51]:


import time

timestamp = int(time.time())

model_package_group_name = f"Summarization-{timestamp}"

print(model_package_group_name)

# In[52]:


inference_image_uri = sagemaker.image_uris.retrieve(
    framework="pytorch",
    region=region,
    version="1.13",
    instance_type=deploy_instance_type,
    image_scope="inference",
)
print(inference_image_uri)

# In[53]:


from sagemaker.workflow.step_collections import RegisterModel

register_step = RegisterModel(
    name="Summarization",
    estimator=estimator,
    image_uri=inference_image_uri,  # we have to specify, by default it's using training image
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/jsonlines"],
    response_types=["application/jsonlines"],
    inference_instances=[deploy_instance_type],
    transform_instances=[deploy_instance_type],
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status,
    model_metrics=model_metrics,
)

# # Create Model for Deployment Step

# In[54]:


from sagemaker.model import Model

model_name = "model-{}".format(timestamp)

model = Model(
    name=model_name,
    image_uri=inference_image_uri,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=sess,
    role=role,
)

# In[55]:


from sagemaker.inputs import CreateModelInput

create_inputs = CreateModelInput(
    instance_type=deploy_instance_type,
)

# In[56]:


from sagemaker.workflow.steps import CreateModelStep

create_step = CreateModelStep(
    name="CreateModel",
    model=model,
    inputs=create_inputs,
)

# # Conditional Deployment Step
# ![](img/pipeline-6.png)
# 
# Finally, we'd like to only register this model if the metrics of the model, as determined by our evaluation step, exceeded a given threshold. A `ConditionStep` allows for pipelines to support conditional execution in the pipeline DAG based on conditions of step properties.
# 
# Below, we do the following:
# * define a condition on the evaluation metrics found in the output of the evaluation step
# * use the condition in the list of conditions in a `ConditionStep`
# * pass the `RegisterModel` step collection into the `if_steps` of the `ConditionStep`

# In[57]:


from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)

min_rouge_value = ParameterFloat(name="MinRouge1Value", default_value=0.005)

min_rouge_condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step=evaluation_step,
        property_file=evaluation_report,
        json_path="metrics.eval_rouge1.value",
    ),
    right=min_rouge_value,  # eval_loss
)

min_rouge_condition_step = ConditionStep(
    name="EvaluationCondition",
    conditions=[min_rouge_condition],
    if_steps=[register_step, create_step],  # success, continue with model registration
    else_steps=[],  # fail, end the pipeline
)

# # Define a Pipeline of Parameters, Steps, and Conditions
# 
# Let's tie it all up into a workflow pipeline so we can execute it, and even schedule it.
# 
# A pipeline requires a `name`, `parameters`, and `steps`. Names must be unique within an `(account, region)` pair so we tack on the timestamp to the name.
# 
# Note:
# 
# * All the parameters used in the definitions must be present.
# * Steps passed into the pipeline need not be in the order of execution. The SageMaker Workflow service will resolve the _data dependency_ DAG as steps the execution complete.
# * Steps must be unique to either pipeline step list or a single condition step if/else list.

# # Submit the Pipeline to SageMaker for Execution 
# 
# Let's submit our pipeline definition to the workflow service. The role passed in will be used by the workflow service to create all the jobs defined in the steps.

# # Create Pipeline

# ### _Ignore any `WARNING` below._

# In[58]:


from sagemaker.workflow.pipeline import Pipeline

existing_pipelines = 0

existing_pipelines_response = sm.list_pipelines(
    PipelineNamePrefix=pipeline_name,
    SortOrder="Descending",
)

if "PipelineSummaries" in existing_pipelines_response.keys():
    if len(existing_pipelines_response["PipelineSummaries"]) > 0:
        existing_pipelines = existing_pipelines + 1
        print("[INFO] You already have created {} pipeline with name {}.".format(existing_pipelines, pipeline_name))
    else:
        pass

if existing_pipelines == 0:  # Only create the pipeline one time
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            processing_instance_count,
            processing_instance_type,
            train_split_percentage,
            validation_split_percentage,
            test_split_percentage,
            train_instance_type,
            train_instance_count,
            epochs,
            learning_rate,
            weight_decay,
            train_sample_percentage,
            train_batch_size,
            validation_batch_size,
            test_batch_size,
            train_volume_size,
            input_mode,
            min_rouge_value,
            model_approval_status,
            deploy_instance_type,
            deploy_instance_count,
            model_checkpoint.to_string(),
        ],
        steps=[processing_step, training_step, evaluation_step, min_rouge_condition_step],
        sagemaker_session=sess,
    )

    pipeline.upsert(role_arn=role)["PipelineArn"]
    print("Created pipeline with name {}".format(pipeline_name))
else:
    print(
        "****************************************************************************************************************"
    )
    print(
        "You have already create a pipeline with the name {}. This is OK. Please continue to the next cell.".format(
            pipeline_name
        )
    )
    print(
        "****************************************************************************************************************"
    )

# ### _Ignore any `WARNING` ^^ above ^^._

# # Start Pipeline

# ### _Ignore any `WARNING` below._

# In[59]:


running_executions = 0
completed_executions = 0

if existing_pipelines > 0:
    existing_pipeline_executions_response = sm.list_pipeline_executions(
        PipelineName=pipeline_name,
        SortOrder="Descending",
    )

    if "PipelineExecutionSummaries" in existing_pipeline_executions_response.keys():
        if len(existing_pipeline_executions_response["PipelineExecutionSummaries"]) > 0:
            execution = existing_pipeline_executions_response["PipelineExecutionSummaries"][0]
            if "PipelineExecutionStatus" in execution:
                if execution["PipelineExecutionStatus"] == "Executing":
                    running_executions = running_executions + 1
                else:
                    completed_executions = completed_executions + 1

            print(
                "[INFO] You have {} Pipeline execution(s) currently running and {} execution(s) completed.".format(
                    running_executions, completed_executions
                )
            )
    else:
        pass
else:
    pass

if running_executions == 0:  # Only allow 1 pipeline execution at a time to limit the resources needed
    execution = pipeline.start()
    running_executions = running_executions + 1
    print("Started pipeline {}.  Ignore any warnings above.".format(pipeline_name))
    print(execution.arn)
else:
    print(
        "********************************************************************************************************************"
    )
    print(
        "You have already launched {} pipeline execution(s).  This is OK.  Please continue to see the next cell.".format(
            running_executions
        )
    )
    print(
        "********************************************************************************************************************"
    )

# ### _Ignore any `WARNING` ^^ above ^^._

# # Wait for the Pipeline to Complete
# 
# ### _This next cell takes about 40 mins.  Please be patient._

# In[60]:


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

# ### _Wait for the Pipeline ^^ Above ^^ to Complete_

# # List Pipeline Execution Steps and Statuses After Completion

# In[61]:


pipeline_execution_status = executions_response[0]["PipelineExecutionStatus"]
pipeline_execution_arn = executions_response[0]["PipelineExecutionArn"]

print("Pipeline execution status {}".format(pipeline_execution_status))
print("Pipeline execution arn {}".format(pipeline_execution_arn))

# In[62]:


from pprint import pprint

steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=pipeline_execution_arn)

pprint(steps)

# # List All Artifacts Generated by the Pipeline

# In[63]:


processing_job_name = None
training_job_name = None

# In[64]:


import time
from sagemaker.lineage.visualizer import LineageTableVisualizer

viz = LineageTableVisualizer(sagemaker.session.Session())

for execution_step in reversed(steps["PipelineExecutionSteps"]):
    print(execution_step)
    # We are doing this because there appears to be a bug of this LineageTableVisualizer handling the Processing Step
    if execution_step["StepName"] == "Processing":
        processing_job_name = execution_step["Metadata"]["ProcessingJob"]["Arn"].split("/")[-1]
        print(processing_job_name)
        display(viz.show(processing_job_name=processing_job_name))
    elif execution_step["StepName"] == "Train":
        training_job_name = execution_step["Metadata"]["TrainingJob"]["Arn"].split("/")[-1]
        print(training_job_name)
        display(viz.show(training_job_name=training_job_name))
    else:
        display(viz.show(pipeline_execution_step=execution_step))
        time.sleep(5)

# ## Add Execution Run as Trial to Experiments

# In[65]:


# -aws-processing-job is the default name assigned by ProcessingJob
processing_job_tc = "{}-aws-processing-job".format(processing_job_name)
print(processing_job_tc)

# In[66]:


response = sm.associate_trial_component(TrialComponentName=processing_job_tc, TrialName=pipeline_trial_name)

# In[67]:


# -aws-training-job is the default name assigned by TrainingJob
training_job_tc = "{}-aws-training-job".format(training_job_name)
print(training_job_tc)

# In[68]:


response = sm.associate_trial_component(TrialComponentName=training_job_tc, TrialName=pipeline_trial_name)

# In[ ]:



