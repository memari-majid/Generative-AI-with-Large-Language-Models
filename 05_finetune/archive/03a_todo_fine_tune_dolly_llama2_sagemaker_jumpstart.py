#!/usr/bin/env python
# coding: utf-8

# # Fine-tune LLaMA 2 models on SageMaker JumpStart

# In[ ]:


%pip install datasets==2.15.0

# ## Deploy Pre-trained Model
# 
# ---
# 
# First we will deploy the Llama-2 model as a SageMaker endpoint. To train/deploy 13B and 70B models, please change model_id to "meta-textgeneration-llama-2-7b" and "meta-textgeneration-llama-2-70b" respectively.
# 
# ---

# In[ ]:


model_id, model_version = "meta-textgeneration-llama-2-7b", "2.*"

# In[ ]:


from sagemaker.jumpstart.model import JumpStartModel

pretrained_model = JumpStartModel(model_id=model_id, model_version=model_version)
pretrained_predictor = pretrained_model.deploy()

# ## Invoke the endpoint
# 
# ---
# Next, we invoke the endpoint with some sample queries. Later, in this notebook, we will fine-tune this model with a custom dataset and carry out inference using the fine-tuned model. We will also show comparison between results obtained via the pre-trained and the fine-tuned models.
# 
# ---

# In[25]:


# def print_response(payload, response):
#     print(payload["inputs"])
#     print(f"> {response[0]['generation']}")
#     print("\n==================================\n")

# In[26]:


# payload = {
#     "inputs": "I believe the meaning of life is",
#     "parameters": {
#         "max_new_tokens": 64,
#         "top_p": 0.9,
#         "temperature": 0.6,
#         "return_full_text": False,
#     },
# }
# try:
#     response = pretrained_predictor.predict(payload, custom_attributes="accept_eula=true")
#     print_response(payload, response)
# except Exception as e:
#     print(e)

# ---
# To learn about additional use cases of pre-trained model, please checkout the notebook [Text completion: Run Llama 2 models in SageMaker JumpStart](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart-foundation-models/llama-2-text-completion.ipynb).
# 
# ---

# ## Dataset preparation for fine-tuning
# 
# ---
# 
# You can fine-tune on the dataset with domain adaptation format or instruction tuning format. Please find more details in the section [Dataset instruction](#Dataset-instruction). In this demo, we will use a subset of [Dolly dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k) in an instruction tuning format. Dolly dataset contains roughly 15,000 instruction following records for various categories such as question answering, summarization, information extraction etc. It is available under Apache 2.0 license. We will select the summarization examples for fine-tuning.
# 
# 
# Training data is formatted in JSON lines (.jsonl) format, where each line is a dictionary representing a single data sample. All training data must be in a single folder, however it can be saved in multiple jsonl files. The training folder can also contain a template.json file describing the input and output formats.
# 
# To train your model on a collection of unstructured dataset (text files), please see the section [Example fine-tuning with Domain-Adaptation dataset format](#Example-fine-tuning-with-Domain-Adaptation-dataset-format) in the Appendix.
# 
# ---

# In[27]:


from datasets import load_dataset

dolly_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# To train for question answering/information extraction, you can replace the assertion in next line to example["category"] == "closed_qa"/"information_extraction".
summarization_dataset = dolly_dataset.filter(lambda example: example["category"] == "summarization")
summarization_dataset = summarization_dataset.remove_columns("category")

# We split the dataset into two where test data is used to evaluate at the end.
train_and_test_dataset = summarization_dataset.train_test_split(test_size=0.1)

# Dumping the training data to a local file to be used for training.
train_and_test_dataset["train"].to_json("train.jsonl")

# In[28]:


train_and_test_dataset["train"][0]

# ---
# Next, we create a prompt template for using the data in an instruction / input format for the training job (since we are instruction fine-tuning the model in this example), and also for inferencing the deployed endpoint.
# 
# ---

# In[ ]:


import json

template = {
    "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n",
    "completion": " {response}",
}
with open("template.json", "w") as f:
    json.dump(template, f)

# ### Upload dataset to S3
# ---
# 
# We will upload the prepared dataset to S3 which will be used for fine-tuning.
# 
# ---

# In[30]:


from sagemaker.s3 import S3Uploader
import sagemaker
import random

output_bucket = sagemaker.Session().default_bucket()
local_data_file = "train.jsonl"
train_data_location = f"s3://{output_bucket}/dolly_dataset"
S3Uploader.upload(local_data_file, train_data_location)
S3Uploader.upload("template.json", train_data_location)
print(f"Training data: {train_data_location}")

# ## Train the model
# ---
# Next, we fine-tune the LLaMA v2 7B model on the summarization dataset from Dolly. Finetuning scripts are based on scripts provided by [this repo](https://github.com/facebookresearch/llama-recipes/tree/main). To learn more about the fine-tuning scripts, please checkout section [5. Few notes about the fine-tuning method](#5.-Few-notes-about-the-fine-tuning-method). For a list of supported hyper-parameters and their default values, please see section [3. Supported Hyper-parameters for fine-tuning](#3.-Supported-Hyper-parameters-for-fine-tuning).
# 
# ---

# In[33]:


from sagemaker.jumpstart.estimator import JumpStartEstimator

estimator = JumpStartEstimator(
    model_id=model_id,
    model_version=model_version,
    environment={"accept_eula": "true"},
    #disable_output_compression=True,  # For Llama-2-70b, add instance_type = "ml.g5.48xlarge"
)

# By default, instruction tuning is set to false. 
# To use instruction tuning dataset you set instruction_tuned=True
estimator.set_hyperparameters(instruction_tuned="True", epoch="5", max_input_length="1024")
estimator.fit({"training": train_data_location})

# Studio Kernel Dying issue:  If your studio kernel dies and you lose reference to the estimator object, please see section [6. Studio Kernel Dead/Creating JumpStart Model from the training Job](#6.-Studio-Kernel-Dead/Creating-JumpStart-Model-from-the-training-Job) on how to deploy endpoint using the training job name and the model id. 
# 

# ### Deploy the fine-tuned model
# ---
# Next, we deploy fine-tuned model. We will compare the performance of fine-tuned and pre-trained model.
# 
# ---

# In[ ]:


finetuned_predictor = estimator.deploy()

# ### Evaluate the pre-trained and fine-tuned model
# ---
# Next, we use the test data to evaluate the performance of the fine-tuned model and compare it with the pre-trained model. 
# 
# ---

# In[ ]:


import pandas as pd
from IPython.display import display, HTML

test_dataset = train_and_test_dataset["test"]

inputs, ground_truth_responses, responses_before_finetuning, responses_after_finetuning = (
    [],
    [],
    [],
    [],
)


def predict_and_print(datapoint):
    # For instruction fine-tuning, we insert a special key between input and output
    input_output_demarkation_key = "\n\n### Response:\n"

    payload = {
        "inputs": template["prompt"].format(
            instruction=datapoint["instruction"], context=datapoint["context"]
        )
        + input_output_demarkation_key,
        "parameters": {"max_new_tokens": 100},
    }
    inputs.append(payload["inputs"])
    ground_truth_responses.append(datapoint["response"])
    # Please change the following line to "accept_eula=True"
    pretrained_response = pretrained_predictor.predict(
        payload, custom_attributes="accept_eula=false"
    )
    responses_before_finetuning.append(pretrained_response[0]["generation"])
    # Please change the following line to "accept_eula=True"
    finetuned_response = finetuned_predictor.predict(payload, custom_attributes="accept_eula=false")
    responses_after_finetuning.append(finetuned_response[0]["generation"])


try:
    for i, datapoint in enumerate(test_dataset.select(range(5))):
        predict_and_print(datapoint)

    df = pd.DataFrame(
        {
            "Inputs": inputs,
            "Ground Truth": ground_truth_responses,
            "Response from non-finetuned model": responses_before_finetuning,
            "Response from fine-tuned model": responses_after_finetuning,
        }
    )
    display(HTML(df.to_html()))
except Exception as e:
    print(e)

# ### Clean up resources

# In[ ]:


# # Delete resources
# pretrained_predictor.delete_model()
# pretrained_predictor.delete_endpoint()
# finetuned_predictor.delete_model()
# finetuned_predictor.delete_endpoint()

# # Appendix

# ### 1. Supported Inference Parameters
# 
# ---
# This model supports the following inference payload parameters:
# 
# * **max_new_tokens:** Model generates text until the output length (excluding the input context length) reaches max_new_tokens. If specified, it must be a positive integer.
# * **temperature:** Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If `temperature` -> 0, it results in greedy decoding. If specified, it must be a positive float.
# * **top_p:** In each step of text generation, sample from the smallest possible set of words with cumulative probability `top_p`. If specified, it must be a float between 0 and 1.
# * **return_full_text:** If True, input text will be part of the output generated text. If specified, it must be boolean. The default value for it is False.
# 
# You may specify any subset of the parameters mentioned above while invoking an endpoint. 
# 
# 
# ### Notes
# - If `max_new_tokens` is not defined, the model may generate up to the maximum total tokens allowed, which is 4K for these models. This may result in endpoint query timeout errors, so it is recommended to set `max_new_tokens` when possible. For 7B, 13B, and 70B models, we recommend to set `max_new_tokens` no greater than 1500, 1000, and 500 respectively, while keeping the total number of tokens less than 4K.
# - In order to support a 4k context length, this model has restricted query payloads to only utilize a batch size of 1. Payloads with larger batch sizes will receive an endpoint error prior to inference.
# 
# ---

# ### 2. Dataset formatting instruction for training
# 
# ---
# 
# ####  Fine-tune the Model on a New Dataset
# We currently offer two types of fine-tuning: instruction fine-tuning and domain adaption fine-tuning. You can easily switch to one of the training 
# methods by specifying parameter `instruction_tuned` being 'True' or 'False'.
# 
# 
# #### 2.1. Domain adaptation fine-tuning
# The Text Generation model can also be fine-tuned on any domain specific dataset. After being fine-tuned on the domain specific dataset, the model
# is expected to generate domain specific text and solve various NLP tasks in that specific domain with **few shot prompting**.
# 
# Below are the instructions for how the training data should be formatted for input to the model.
# 
# - **Input:** A train and an optional validation directory. Each directory contains a CSV/JSON/TXT file. 
#   - For CSV/JSON files, the train or validation data is used from the column called 'text' or the first column if no column called 'text' is found.
#   - The number of files under train and validation (if provided) should equal to one, respectively. 
# - **Output:** A trained model that can be deployed for inference. 
# 
# Below is an example of a TXT file for fine-tuning the Text Generation model. The TXT file is SEC filings of Amazon from year 2021 to 2022.
# 
# ```Note About Forward-Looking Statements
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
# GENERAL
# Embracing Our Future ...
# ```
# 
# 
# #### 2.2. Instruction fine-tuning
# The Text generation model can be instruction-tuned on any text data provided that the data 
# is in the expected format. The instruction-tuned model can be further deployed for inference. 
# Below are the instructions for how the training data should be formatted for input to the 
# model.
# 
# Below are the instructions for how the training data should be formatted for input to the model.
# 
# - **Input:** A train and an optional validation directory. Train and validation directories should contain one or multiple JSON lines (`.jsonl`) formatted files. In particular, train directory can also contain an optional `*.json` file describing the input and output formats. 
#   - The best model is selected according to the validation loss, calculated at the end of each epoch.
#   If a validation set is not given, an (adjustable) percentage of the training data is
#   automatically split and used for validation.
#   - The training data must be formatted in a JSON lines (`.jsonl`) format, where each line is a dictionary
# representing a single data sample. All training data must be in a single folder, however
# it can be saved in multiple jsonl files. The `.jsonl` file extension is mandatory. The training
# folder can also contain a `template.json` file describing the input and output formats. If no
# template file is given, the following template will be used:
#   ```json
#   {
#     "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}",
#     "completion": "{response}"
#   }
#   ```
#   - In this case, the data in the JSON lines entries must include `instruction`, `context` and `response` fields. If a custom template is provided it must also use `prompt` and `completion` keys to define
#   the input and output templates.
#   Below is a sample custom template:
# 
#   ```json
#   {
#     "prompt": "question: {question} context: {context}",
#     "completion": "{answer}"
#   }
#   ```
# Here, the data in the JSON lines entries must include `question`, `context` and `answer` fields. 
# - **Output:** A trained model that can be deployed for inference. 
# 
# ---

# #### 2.3. Example fine-tuning with Domain-Adaptation dataset format
# ---
# We provide a subset of SEC filings data of Amazon in domain adaptation dataset format. It is downloaded from publicly available [EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch). Instruction of accessing the data is shown [here](https://www.sec.gov/os/accessing-edgar-data).
# 
# License: [Creative Commons Attribution-ShareAlike License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/legalcode).
# 
# Please uncomment the following code to fine-tune the model on dataset in domain adaptation format.
# 
# ---

# In[ ]:


import boto3
model_id = "meta-textgeneration-llama-2-7b"

estimator = JumpStartEstimator(model_id=model_id,  environment={"accept_eula": "true"},instance_type = "ml.g5.24xlarge")
estimator.set_hyperparameters(instruction_tuned="False", epoch="5")
estimator.fit({"training": f"s3://jumpstart-cache-prod-{boto3.Session().region_name}/training-datasets/sec_amazon"})

# ### 3. Supported Hyper-parameters for fine-tuning
# ---
# - epoch: The number of passes that the fine-tuning algorithm takes through the training dataset. Must be an integer greater than 1. Default: 5
# - learning_rate: The rate at which the model weights are updated after working through each batch of training examples. Must be a positive float greater than 0. Default: 1e-4.
# - instruction_tuned: Whether to instruction-train the model or not. Must be 'True' or 'False'. Default: 'False'
# - per_device_train_batch_size: The batch size per GPU core/CPU for training. Must be a positive integer. Default: 4.
# - per_device_eval_batch_size: The batch size per GPU core/CPU for evaluation. Must be a positive integer. Default: 1
# - max_train_samples: For debugging purposes or quicker training, truncate the number of training examples to this value. Value -1 means using all of training samples. Must be a positive integer or -1. Default: -1. 
# - max_val_samples: For debugging purposes or quicker training, truncate the number of validation examples to this value. Value -1 means using all of validation samples. Must be a positive integer or -1. Default: -1. 
# - max_input_length: Maximum total input sequence length after tokenization. Sequences longer than this will be truncated. If -1, max_input_length is set to the minimum of 1024 and the maximum model length defined by the tokenizer. If set to a positive value, max_input_length is set to the minimum of the provided value and the model_max_length defined by the tokenizer. Must be a positive integer or -1. Default: -1. 
# - validation_split_ratio: If validation channel is none, ratio of train-validation split from the train data. Must be between 0 and 1. Default: 0.2. 
# - train_data_split_seed: If validation data is not present, this fixes the random splitting of the input training data to training and validation data used by the algorithm. Must be an integer. Default: 0.
# - preprocessing_num_workers: The number of processes to use for the preprocessing. If None, main process is used for preprocessing. Default: "None"
# - lora_r: Lora R. Must be a positive integer. Default: 8.
# - lora_alpha: Lora Alpha. Must be a positive integer. Default: 32
# - lora_dropout: Lora Dropout. must be a positive float between 0 and 1. Default: 0.05. 
# - int8_quantization: If True, model is loaded with 8 bit precision for training. Default for 7B/13B: False. Default for 70B: True.
# - enable_fsdp: If True, training uses Fully Sharded Data Parallelism. Default for 7B/13B: True. Default for 70B: False.
# 
# Note 1: int8_quantization is not supported with FSDP. Also, int8_quantization = 'False' and enable_fsdp = 'False' is not supported due to CUDA memory issues for any of the g5 family instances. Thus, we recommend setting exactly one of int8_quantization or enable_fsdp to be 'True'
# Note 2: Due to the size of the model, 70B model can not be fine-tuned with enable_fsdp = 'True' for any of the supported instance types.
# 
# ---

# ### 4. Supported Instance types
# 
# ---
# We have tested our scripts on the following instances types:
# 
# - 7B: ml.g5.12xlarge, nl.g5.24xlarge, ml.g5.48xlarge, ml.p3dn.24xlarge
# - 13B: ml.g5.24xlarge, ml.g5.48xlarge, ml.p3dn.24xlarge
# - 70B: ml.g5.48xlarge
# 
# Other instance types may also work to fine-tune. Note: When using p3 instances, training will be done with 32 bit precision as bfloat16 is not supported on these instances. Thus, training job would consume double the amount of CUDA memory when training on p3 instances compared to g5 instances.
# 
# ---

# ### 5. Few notes about the fine-tuning method
# 
# ---
# - Fine-tuning scripts are based on [this repo](https://github.com/facebookresearch/llama-recipes/tree/main). 
# - Instruction tuning dataset is first converted into domain adaptation dataset format before fine-tuning. 
# - Fine-tuning scripts utilize Fully Sharded Data Parallel (FSDP) as well as Low Rank Adaptation (LoRA) method fine-tuning the models
# 
# ---

# ### 6. Studio Kernel Dead/Creating JumpStart Model from the training Job
# ---
# Due to the size of the Llama 70B model, training job may take several hours and the studio kernel may die during the training phase. However, during this time, training is still running in SageMaker. If this happens, you can still deploy the endpoint using the training job name with the following code:
# 
# How to find the training job name? Go to Console -> SageMaker -> Training -> Training Jobs -> Identify the training job name and substitute in the following cell. 
# 
# ---

# In[ ]:


# from sagemaker.jumpstart.estimator import JumpStartEstimator
# training_job_name = <<training_job_name>>

# attached_estimator = JumpStartEstimator.attach(training_job_name, model_id)
# attached_estimator.logs()
# attached_estimator.deploy()
