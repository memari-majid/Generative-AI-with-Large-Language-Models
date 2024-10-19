#!/usr/bin/env python
# coding: utf-8

# # Introduction to SageMaker JumpStart - Text Generation with Mistral models
# 
# ---
# In this demo notebook, we demonstrate how to use the SageMaker Python SDK to fine-tuning and deploy [Mistral 7B](mistralai/Mistral-7B-v0.1) models for text generation. For fine-tuning, we include two types of fine-tuning: instruction fine-tuning and domain adaption fine-tuning.  
# 
# ---

# Below is the content of the notebook.
# 
# 1. [Instruction fine-tuning](#1.-Instruction-fine-tuning)
#    * [1.1. Preparing training data](#1.1.-Preparing-training-data)
#    * [1.2. Prepare training parameters](#1.2.-Prepare-training-parameters)
#    * [1.3. Starting training](#1.3.-Starting-training)
#    * [1.4. Deploying inference endpoints](#1.4.-Deploying-inference-endpoints)
#    * [1.5. Clean up endpoint](#1.6.-Clean-up-the-endpoint)
# 2. [Domain adaptation fine-tuning](#2.-Domain-adaptation-fine-tuning)
#    * [2.1. Preparing training data](#2.1.-Preparing-training-data)
#    * [2.2. Prepare training parameters](#2.2.-Prepare-training-parameters)
#    * [2.3. Starting training](#2.3.-Starting-training)
#    * [2.4. Deploying inference endpoints](#2.4.-Deploying-inference-endpoints)
#    * [2.5. Running inference queries and compare model performances](#2.5.-Running-inference-queries-and-compare-model-performances)
#    * [2.6. Clean up endpoint](#2.6.-Clean-up-the-endpoint)

# Install latest SageMaker and dependencies.

# In[2]:


!pip install sagemaker --quiet --upgrade --force-reinstall
!pip install ipywidgets==7.0.0 --quiet
!pip install datasets --quiet

# ## 1. Instruction fine-tuning
# 
# Now, we demonstrate how to instruction-tune `huggingface-llm-mistral-7b` model for a new task. The Mistral-7B-v0.1 Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters. Mistral-7B-v0.1 outperforms Llama 2 13B on all benchmarks we tested. For details, see its [HuggingFace webpage](https://huggingface.co/mistralai/Mistral-7B-v0.1).

# In[3]:


model_id, model_version = "huggingface-llm-mistral-7b", "*"

# ### 1.1. Preparing training data
# 
# You can fine-tune on the dataset with domain adaptation format or instruction tuning format. In this section, we will use a subset of [Dolly dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k) in an instruction tuning format. Dolly dataset contains roughly 15,000 instruction following records for various categories such as question answering, summarization, information extraction etc. It is available under Apache 2.0 license. We will select the summarization examples for fine-tuning.
# 
# Training data is formatted in JSON lines (.jsonl) format, where each line is a dictionary representing a single data sample. All training data must be in a single folder, however it can be saved in multiple jsonl files. The training folder can also contain a template.json file describing the input and output formats.

# In[4]:


import boto3
import sagemaker
import json

# Get current region, role, and default bucket
aws_region = boto3.Session().region_name
aws_role = sagemaker.session.Session().get_caller_identity_arn()
output_bucket = sagemaker.Session().default_bucket()

# This will be useful for printing
newline, bold, unbold = "\n", "\033[1m", "\033[0m"

print(f"{bold}aws_region:{unbold} {aws_region}")
print(f"{bold}aws_role:{unbold} {aws_role}")
print(f"{bold}output_bucket:{unbold} {output_bucket}")

# In[5]:


from datasets import load_dataset

dolly_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# To train for question answering/information extraction, you can replace the assertion in next line to example["category"] == "closed_qa"/"information_extraction".
summarization_dataset = dolly_dataset.filter(lambda example: example["category"] == "summarization")
summarization_dataset = summarization_dataset.remove_columns("category")

# We split the dataset into two where test data is used to evaluate at the end.
train_and_test_dataset = summarization_dataset.train_test_split(test_size=0.1)

# Dumping the training data to a local file to be used for training.
train_and_test_dataset["train"].to_json("train.jsonl")

# In[6]:


train_and_test_dataset["train"][0]

# The training data must be formatted in JSON lines (.jsonl) format, where each line is a dictionary representing a single data sample. All training data must be in a single folder, however it can be saved in multiple jsonl files. The .jsonl file extension is mandatory. The training folder can also contain a template.json file describing the input and output formats.
# 
# If no template file is given, the following default template will be used:
# 
# ```json
# {
#     "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}`,
#     "completion": "{response}",
# }
# ```
# 
# In this case, the data in the JSON lines entries must include `instruction`, `context`, and `response` fields.
# 
# Different from using the default prompt template, in this demo we are going to use a custom template (see below).

# In[7]:


import json

template = {
    "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n",
    "completion": " {response}",
}
with open("template.json", "w") as f:
    json.dump(template, f)

# Next, we are going to reformat the SQuAD 2.0 dataset. The processed data is saved as `task-data.jsonl` file. Given the prompt template defined in above cell, each entry in the `task-data.jsonl` file include **`context`** and **`question`** fields. For demonstration purpose, we limit the number of training examples to be 2000.

# In[8]:


from sagemaker.s3 import S3Uploader
import sagemaker
import random

output_bucket = sagemaker.Session().default_bucket()
local_data_file = "train.jsonl"
train_data_location = f"s3://{output_bucket}/dolly_dataset_mistral"
S3Uploader.upload(local_data_file, train_data_location)
S3Uploader.upload("template.json", train_data_location)
print(f"Training data: {train_data_location}")

# Upload the prompt template (`template.json`) and training data (`task-data.jsonl`) into S3 bucket.

# ### 1.2. Prepare training parameters

# In[9]:


from sagemaker import hyperparameters

my_hyperparameters = hyperparameters.retrieve_default(
    model_id=model_id, model_version=model_version
)
print(my_hyperparameters)

# Overwrite the hyperparameters. **Note. You can select the LoRA method for your fine-tuning by selecting peft_type=`lora` in the hyper-parameters.**

# In[10]:


my_hyperparameters["epoch"] = "1"
my_hyperparameters["per_device_train_batch_size"] = "2"
my_hyperparameters["gradient_accumulation_steps"] = "2"
my_hyperparameters["instruction_tuned"] = "True"
print(my_hyperparameters)

# Validate hyperparameters

# In[11]:


hyperparameters.validate(
    model_id=model_id, model_version=model_version, hyperparameters=my_hyperparameters
)

# ### 1.3. Starting training

# In[12]:


from sagemaker.jumpstart.estimator import JumpStartEstimator

instruction_tuned_estimator = JumpStartEstimator(
    model_id=model_id,
    hyperparameters=my_hyperparameters,
    instance_type="ml.g5.12xlarge",
)
instruction_tuned_estimator.fit({"train": train_data_location}, logs=True)

# Extract Training performance metrics. Performance metrics such as training loss and validation accuracy/loss can be accessed through cloudwatch while the training. We can also fetch these metrics and analyze them within the notebook.

# In[ ]:


from sagemaker import TrainingJobAnalytics

training_job_name = instruction_tuned_estimator.latest_training_job.job_name

df = TrainingJobAnalytics(training_job_name=training_job_name).dataframe()
df.head(10)

# ### 1.4. Deploying inference endpoints

# In[ ]:


instruction_tuned_predictor = instruction_tuned_estimator.deploy()

# Note. For dolly dataset, we observe the performance of fine-tuned model is equivalently excellent to that of pre-trained model. This is likely due to the Mistral 7B has already learned knowledge in this domain. The code example above is just a demonstration on how to fine-tune such model in an instruction way. For your own use case, please substitute the example dolly dataset by yours.

# ### 1.6. Clean up the endpoint

# In[ ]:


# Delete the SageMaker endpoint
instruction_tuned_predictor.delete_model()
instruction_tuned_predictor.delete_endpoint()

# ## 2. Domain adaptation fine-tuning
# 
# We also have domain adaptation fine-tuning enabled for Mistral models. Different from instruction fine-tuning, you do not need prepare instruction-formatted dataset and can directly use unstructured text document which is demonstrated as below. However, the model that is domain-adaptation fine-tuned may not give concise responses as the instruction-tuned model because of less restrictive requirements on training data formats.

# We will use financial text from SEC filings to fine tune Mistral 7B model for financial applications. 
# 
# Here are the requirements for train and validation data.
# 
# - **Input**: A train and an optional validation directory. Each directory contains a CSV/JSON/TXT file.
#     - For CSV/JSON files, the train or validation data is used from the column called 'text' or the first column if no column called 'text' is found.
#     - The number of files under train and validation (if provided) should equal to one.
# - **Output**: A trained model that can be deployed for inference.
# 
# Below is an example of a TXT file for fine-tuning the Text Generation model. The TXT file is SEC filings of Amazon from year 2021 to 2022.
# 
# ---
# ```
# This report includes estimates, projections, statements relating to our
# business plans, objectives, and expected operating results that are “forward-
# looking statements” within the meaning of the Private Securities Litigation
# Reform Act of 1995, Section 27A of the Securities Act of 1933, and Section 21E
# of the Securities Exchange Act of 1934. Forward-looking statements may appear
# throughout this report, including the following sections: “Business” (Part I,
# Item 1 of this Form 10-K), “Risk Factors” (Part I, Item 1A of this Form 10-K),
# and “Management’s Discussion and Analysis of Financial Condition and Results
# of Operations” (Part II, Item 7 of this Form 10-K). These forward-looking
# statements generally are identified by the words “believe,” “project,”
# “expect,” “anticipate,” “estimate,” “intend,” “strategy,” “future,”
# “opportunity,” “plan,” “may,” “should,” “will,” “would,” “will be,” “will
# continue,” “will likely result,” and similar expressions. Forward-looking
# statements are based on current expectations and assumptions that are subject
# to risks and uncertainties that may cause actual results to differ materially.
# We describe risks and uncertainties that could cause actual results and events
# to differ materially in “Risk Factors,” “Management’s Discussion and Analysis
# of Financial Condition and Results of Operations,” and “Quantitative and
# Qualitative Disclosures about Market Risk” (Part II, Item 7A of this Form
# 10-K). Readers are cautioned not to place undue reliance on forward-looking
# statements, which speak only as of the date they are made. We undertake no
# obligation to update or revise publicly any forward-looking statements,
# whether because of new information, future events, or otherwise.
# 
# ...
# ```
# ---
# SEC filings data of Amazon is downloaded from publicly available [EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch). Instruction of accessing the data is shown [here](https://www.sec.gov/os/accessing-edgar-data).

# ### 2.1. Preparing training data
# 
# The training data of SEC filing of Amazon has been pre-saved in the S3 bucket.

# In[ ]:


from sagemaker.jumpstart.utils import get_jumpstart_content_bucket

# Sample training data is available in this bucket
data_bucket = get_jumpstart_content_bucket(aws_region)
data_prefix = "training-datasets/sec_data"

training_dataset_s3_path = f"s3://{data_bucket}/{data_prefix}/train/"
validation_dataset_s3_path = f"s3://{data_bucket}/{data_prefix}/validation/"

# ### 2.2. Prepare training parameters
# 
# We pick the `max_input_length` to be 2048 on `g5.12xlarge`. You can use higher input length on larger instance type.

# In[ ]:


from sagemaker import hyperparameters

my_hyperparameters = hyperparameters.retrieve_default(
    model_id=model_id, model_version=model_version
)

my_hyperparameters["epoch"] = "3"
my_hyperparameters["per_device_train_batch_size"] = "2"
my_hyperparameters["instruction_tuned"] = "False"
my_hyperparameters["max_input_length"] = "2048"
print(my_hyperparameters)

# Validate hyperparameters

# In[ ]:


hyperparameters.validate(
    model_id=model_id, model_version=model_version, hyperparameters=my_hyperparameters
)

# ### 2.3. Starting training

# In[ ]:


from sagemaker.jumpstart.estimator import JumpStartEstimator

domain_adaptation_estimator = JumpStartEstimator(
    model_id=model_id,
    hyperparameters=my_hyperparameters,
    instance_type="ml.g5.12xlarge",
)
domain_adaptation_estimator.fit(
    {"train": training_dataset_s3_path, "validation": validation_dataset_s3_path}, logs=True
)

# Extract Training performance metrics. Performance metrics such as training loss and validation accuracy/loss can be accessed through cloudwatch while the training. We can also fetch these metrics and analyze them within the notebook

# In[ ]:


from sagemaker import TrainingJobAnalytics

training_job_name = domain_adaptation_estimator.latest_training_job.job_name

df = TrainingJobAnalytics(training_job_name=training_job_name).dataframe()
df.head(10)

# ### 2.4. Deploying inference endpoints
# 
# We deploy the domain-adaptation fine-tuned and pretrained models separately, and compare their performances.
# 
# We firstly deploy the domain-adaptation fine-tuned model.

# In[ ]:


domain_adaptation_predictor = domain_adaptation_estimator.deploy()

# Next, we deploy the pre-trained `huggingface-llm-mistral-7b`. 

# In[ ]:


from sagemaker.jumpstart.model import JumpStartModel

my_model = JumpStartModel(model_id=model_id)
pretrained_predictor = my_model.deploy()

# ### 2.5. Running inference queries and compare model performances

# In[ ]:


parameters = {
    "max_new_tokens": 300,
    "top_k": 50,
    "top_p": 0.8,
    "do_sample": True,
    "temperature": 1,
}


def query_endpoint_with_json_payload(encoded_json, endpoint_name):
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=encoded_json
    )
    return response


def parse_response(query_response):
    model_predictions = json.loads(query_response["Body"].read())
    return model_predictions[0]["generated_text"]


def generate_response(endpoint_name, text):
    payload = {"inputs": f"{text}:", "parameters": parameters}
    query_response = query_endpoint_with_json_payload(
        json.dumps(payload).encode("utf-8"), endpoint_name=endpoint_name
    )
    generated_texts = parse_response(query_response)
    print(f"Response: {generated_texts}{newline}")

# In[ ]:


newline, bold, unbold = "\n", "\033[1m", "\033[0m"

test_paragraph_domain_adaption = [
    "This Form 10-K report shows that",
    "We serve consumers through",
    "Our vision is",
]


for paragraph in test_paragraph_domain_adaption:
    print("-" * 80)
    print(paragraph)
    print("-" * 80)
    print(f"{bold}pre-trained{unbold}")
    generate_response(pretrained_predictor.endpoint_name, paragraph)
    print(f"{bold}fine-tuned{unbold}")
    generate_response(domain_adaptation_predictor.endpoint_name, paragraph)

# As you can, the fine-tuned model starts to generate responses that are more specific to the domain of fine-tuning data which is relating to SEC report of Amazon.

# ### 2.6. Clean up the endpoint

# In[ ]:


# Delete the SageMaker endpoint
pretrained_predictor.delete_model()
pretrained_predictor.delete_endpoint()
domain_adaptation_predictor.delete_model()
domain_adaptation_predictor.delete_endpoint()
