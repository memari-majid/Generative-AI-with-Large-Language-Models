#!/usr/bin/env python
# coding: utf-8

# # Introduction to SageMaker JumpStart - Text Generation with Falcon models
# 
# ---
# In this demo notebook, we demonstrate how to use the SageMaker Python SDK to deploy and fine-tuning Falcon models for text generation. For inference, we show several example use cases including code generation, question answering, translation etc. For fine-tuning, we include two types of fine-tuning: instruction fine-tuning and domain adaption fine-tuning. 
# 
# The Falcon model is a permissively licensed ([Apache-2.0](https://jumpstart-cache-prod-us-east-2.s3.us-east-2.amazonaws.com/licenses/Apache-License/LICENSE-2.0.txt)) open source model trained on the [RefinedWeb dataset](https://huggingface.co/datasets/tiiuae/falcon-refinedweb). 
# 
# ---

# Below is the content of the notebook.
# 
# 1. [Deploy Falcon model for inference](#1.-Deploying-Falcon-model-for-inference)
#    * [1.1. Changing instance type](#1.1.-Changing-instance-type)
#    * [1.2. Changing number of GPUs](#1.2.-Changing-number-of-GPUs)
#    * [1.3. About the model](#1.3.-About-the-model)
#    * [1.4. Supported parameters](#1.4.-Supported-parameters)
# 2. [Instruction fine-tuning](#2.-Instruction-fine-tuning)
#    * [2.1. Preparing training data](#2.1.-Preparing-training-data)
#    * [2.2. Prepare training parameters](#2.2.-Prepare-training-parameters)
#    * [2.3. Starting training](#2.3.-Starting-training)
#    * [2.4. Deploying inference endpoints](#2.4.-Deploying-inference-endpoints)
#    * [2.5. Running inference queries and compare model performances](#2.5.-Running-inference-queries-and-compare-model-performances)
#    * [2.6. Clean up endpoint](#2.6.-Clean-up-the-endpoint)
# 3. [Domain adaptation fine-tuning](#3.-Domain-adaptation-fine-tuning)
#    * [3.1. Preparing training data](#3.1.-Preparing-training-data)
#    * [3.2. Prepare training parameters](#3.2.-Prepare-training-parameters)
#    * [3.3. Starting training](#3.3.-Starting-training)
#    * [3.4. Deploying inference endpoints](#3.4.-Deploying-inference-endpoints)
#    * [3.5. Running inference queries and compare model performances](#3.5.-Running-inference-queries-and-compare-model-performances)
#    * [3.6. Clean up endpoint](#3.6.-Clean-up-the-endpoint)

# ## 1. Deploying Falcon model for inference

# In[2]:


# !pip install sagemaker --quiet --upgrade --force-reinstall
# !pip install ipywidgets==7.0.0 --quiet

# In[3]:


model_id, model_version = "huggingface-llm-falcon-40b-instruct-bf16", "*"

# In[4]:


%%time
from sagemaker.jumpstart.model import JumpStartModel

my_model = JumpStartModel(model_id=model_id)
predictor = my_model.deploy()

# ### 1.1. Changing instance type
# ---
# 
# 
# Models have been tested on the following instance types:
# 
#  - Falcon 7B and 7B instruct: `ml.g5.2xlarge`, `ml.g5.2xlarge`, `ml.g5.4xlarge`, `ml.g5.8xlarge`, `ml.g5.16xlarge`, `ml.g5.12xlarge`, `ml.g5.24xlarge`, `ml.g5.48xlarge`, `ml.p4d.24xlarge`
#  - Falcon 40B and 40B instruct: `ml.g5.12xlarge`, `ml.g5.48xlarge`, `ml.p4d.24xlarge`
# 
# If an instance type is not available in you region, please try a different instance. You can do so by specifying instance type in the JumpStartModel class.
# 
# `my_model = JumpStartModel(model_id="huggingface-llm-falcon-40b-instruct-bf16", instance_type="ml.g5.12xlarge")`
# 
# ---

# ### 1.2. Changing number of GPUs
# ---
# Falcon models are served with HuggingFace (HF) LLM DLC which requires specifying number of GPUs during model deployment. 
# 
# **Falcon 7B and 7B instruct:** HF LLM DLC currently does not support sharding for 7B model. Thus, even if more than one GPU is available on the instance, please do not increase number of GPUs. 
# 
# **Falcon 40B and 40B instruct:** By default number of GPUs are set to 4. However, if you are using `ml.g5.48xlarge` or `ml.p4d.24xlarge`, you can increase number of GPUs to be 8 as follows: 
# 
# `my_model = JumpStartModel(model_id="huggingface-llm-falcon-40b-instruct-bf16", instance_type="ml.g5.48xlarge")`
# 
# `my_model.env['SM_NUM_GPUS'] = '8'`
# 
# `predictor = my_model.deploy()`
# 
# 
# ---

# In[5]:


%%time

prompt = "Tell me about Amazon SageMaker."

payload = {
    "inputs": prompt,
    "parameters": {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_new_tokens": 1024,
        "stop": ["<|endoftext|>", "</s>"]
    }
}

response = predictor.predict(payload)
print(response[0]["generated_text"])

# ### 1.3. About the model
# 
# ---
# Falcon is a causal decoder-only model built by [Technology Innovation Institute](https://www.tii.ae/) (TII) and trained on more than 1 trillion tokens of RefinedWeb enhanced with curated corpora. It was built using custom-built tooling for data pre-processing and model training built on Amazon SageMaker. As of June 6, 2023, it is the best open-source model currently available. Falcon-40B outperforms LLaMA, StableLM, RedPajama, MPT, etc. To see comparison, see [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). It features an architecture optimized for inference, with FlashAttention and multiquery. 
# 
# 
# [Refined Web Dataset](https://huggingface.co/datasets/tiiuae/falcon-refinedweb): Falcon RefinedWeb is a massive English web dataset built by TII and released under an Apache 2.0 license. It is a highly filtered dataset with large scale de-duplication of CommonCrawl. It is observed that models trained on RefinedWeb achieve performance equal to or better than performance achieved by training model on curated datasets, while only relying on web data.
# 
# **Model Sizes:**
# - **Falcon-7b**: It is a 7 billion parameter model trained on 1.5 trillion tokens. It outperforms comparable open-source models (e.g., MPT-7B, StableLM, RedPajama etc.). To see comparison, see [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). To use this model, please select `model_id` in the cell above to be "huggingface-llm-falcon-7b-bf16".
# - **Falcon-40B**: It is a 40 billion parameter model trained on 1 trillion tokens.  It has surpassed renowned models like LLaMA-65B, StableLM, RedPajama and MPT on the public leaderboard maintained by Hugging Face, demonstrating its exceptional performance without specialized fine-tuning. To see comparison, see [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). 
# 
# **Instruct models (Falcon-7b-instruct/Falcon-40B-instruct):** Instruct models are base falcon models fine-tuned on a mixture of chat and instruction datasets. They are ready-to-use chat/instruct models.  To use these models, please select `model_id` in the cell above to be "huggingface-textgeneration-falcon-7b-instruct-bf16" or "huggingface-textgeneration-falcon-40b-instruct-bf16".
# 
# It is [recommended](https://huggingface.co/tiiuae/falcon-7b) that Instruct models should be used without fine-tuning and base models should be fine-tuned further on the specific task.
# 
# **Limitations:**
# 
# - Falcon models are mostly trained on English data and may not generalize to other languages. 
# - Falcon carries the stereotypes and biases commonly encountered online and in the training data. Hence, it is recommended to develop guardrails and to take appropriate precautions for any production use. This is a raw, pretrained model, which should be further finetuned for most usecases.
# 
# 
# ---

# In[6]:


def query_endpoint(payload):
    """Query endpoint and print the response"""
    response = predictor.predict(payload)
    print(f"\033[1m Input:\033[0m {payload['inputs']}")
    print(f"\033[1m Output:\033[0m {response[0]['generated_text']}")

# In[7]:


# Code generation
payload = {"inputs": "Write a program to compute factorial in python:", "parameters":{"max_new_tokens": 200}}
query_endpoint(payload)

# In[8]:


payload = {
    "inputs": "Building a website can be done in 10 simple steps:",
    "parameters":{
        "max_new_tokens": 110,
        "no_repeat_ngram_size": 3
        }
}
query_endpoint(payload)

# In[9]:


# Translation
payload = {
    "inputs": """Translate English to French:

    sea otter => loutre de mer

    peppermint => menthe poivrée

    plush girafe => girafe peluche

    cheese =>""",
    "parameters":{
        "max_new_tokens": 3
    }
}

query_endpoint(payload)

# In[10]:


# Sentiment-analysis
payload = {
    "inputs": """"I hate it when my phone battery dies."
                Sentiment: Negative
                ###
                Tweet: "My day has been :+1:"
                Sentiment: Positive
                ###
                Tweet: "This is the link to the article"
                Sentiment: Neutral
                ###
                Tweet: "This new music video was incredibile"
                Sentiment:""",
    "parameters": {
        "max_new_tokens":2
    }
}
query_endpoint(payload)

# In[11]:


# Question answering
payload = {
    "inputs": "Could you remind me when was the C programming language invented?",
    "parameters":{
        "max_new_tokens": 50
    }
}
query_endpoint(payload)

# In[12]:


# Recipe generation
payload = {"inputs": "What is the recipe for a delicious lemon cheesecake?", "parameters":{"max_new_tokens": 400}}
query_endpoint(payload)

# In[13]:


# Summarization

payload = {
    "inputs":"""Starting today, the state-of-the-art Falcon 40B foundation model from Technology
    Innovation Institute (TII) is available on Amazon SageMaker JumpStart, SageMaker's machine learning (ML) hub
    that offers pre-trained models, built-in algorithms, and pre-built solution templates to help you quickly get
    started with ML. You can deploy and use this Falcon LLM with a few clicks in SageMaker Studio or
    programmatically through the SageMaker Python SDK.
    Falcon 40B is a 40-billion-parameter large language model (LLM) available under the Apache 2.0 license that
    ranked #1 in Hugging Face Open LLM leaderboard, which tracks, ranks, and evaluates LLMs across multiple
    benchmarks to identify top performing models. Since its release in May 2023, Falcon 40B has demonstrated
    exceptional performance without specialized fine-tuning. To make it easier for customers to access this
    state-of-the-art model, AWS has made Falcon 40B available to customers via Amazon SageMaker JumpStart.
    Now customers can quickly and easily deploy their own Falcon 40B model and customize it to fit their specific
    needs for applications such as translation, question answering, and summarizing information.
    Falcon 40B are generally available today through Amazon SageMaker JumpStart in US East (Ohio),
    US East (N. Virginia), US West (Oregon), Asia Pacific (Tokyo), Asia Pacific (Seoul), Asia Pacific (Mumbai),
    Europe (London), Europe (Frankfurt), Europe (Ireland), and Canada (Central),
    with availability in additional AWS Regions coming soon. To learn how to use this new feature,
    please see SageMaker JumpStart documentation, the Introduction to SageMaker JumpStart –
    Text Generation with Falcon LLMs example notebook, and the blog Technology Innovation Institute trainsthe
    state-of-the-art Falcon LLM 40B foundation model on Amazon SageMaker. Summarize the article above:""",
    "parameters":{
        "max_new_tokens":200
        }
    }
query_endpoint(payload)

# ### 1.4. Supported parameters
# 
# ***
# Some of the supported parameters while performing inference are the following:
# 
# * **max_length:** Model generates text until the output length (which includes the input context length) reaches `max_length`. If specified, it must be a positive integer.
# * **max_new_tokens:** Model generates text until the output length (excluding the input context length) reaches `max_new_tokens`. If specified, it must be a positive integer.
# * **num_beams:** Number of beams used in the greedy search. If specified, it must be integer greater than or equal to `num_return_sequences`.
# * **no_repeat_ngram_size:** Model ensures that a sequence of words of `no_repeat_ngram_size` is not repeated in the output sequence. If specified, it must be a positive integer greater than 1.
# * **temperature:** Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If `temperature` -> 0, it results in greedy decoding. If specified, it must be a positive float.
# * **early_stopping:** If True, text generation is finished when all beam hypotheses reach the end of sentence token. If specified, it must be boolean.
# * **do_sample:** If True, sample the next word as per the likelihood. If specified, it must be boolean.
# * **top_k:** In each step of text generation, sample from only the `top_k` most likely words. If specified, it must be a positive integer.
# * **top_p:** In each step of text generation, sample from the smallest possible set of words with cumulative probability `top_p`. If specified, it must be a float between 0 and 1.
# * **return_full_text:** If True, input text will be part of the output generated text. If specified, it must be boolean. The default value for it is False.
# * **stop**: If specified, it must a list of strings. Text generation stops if any one of the specified strings is generated.
# 
# We may specify any subset of the parameters mentioned above while invoking an endpoint. 
# 
# For more parameters and information on HF LLM DLC, please see [this article](https://huggingface.co/blog/sagemaker-huggingface-llm#4-run-inference-and-chat-with-our-model).
# ***

# ## 2. Instruction fine-tuning
# 
# Now, we demonstrate how to instruction-tune `huggingface-llm-falcon-7b-instruct-bf16` model for a new task. As mentioned in [Section 1.3 About the models](#1.3.-About-the-model),  **Falcon-7b-instruct**/**Falcon-40B-instruct** models are instruction base falcon models fine-tuned on a mixture of chat and instruction datasets. 
# 
# **In this task, given a piece of context, the model is asked to generate questions that are relevant to the text, but `cannot` be answered based on provided information. Examples are given in the inference section of this notebook.**

# ### 2.1. Preparing training data
# We will use a subset of SQuAD2.0 for supervised fine-tuning. This dataset contains questions posed by human annotators on a set of Wikipedia articles. In addition to questions with answers, SQuAD2.0 contains about 50k unanswerable questions. Such questions are plausible, but cannot be directly answered from the articles' content. We only use unanswerable questions for our task.
# 
# Citation: @article{rajpurkar2018know, title={Know what you don't know: Unanswerable questions for SQuAD}, author={Rajpurkar, Pranav and Jia, Robin and Liang, Percy}, journal={arXiv preprint arXiv:1806.03822}, year={2018} }
# 
# License: Creative Commons Attribution-ShareAlike License (CC BY-SA 4.0)

# In[14]:


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

# In[15]:


from sagemaker.s3 import S3Downloader

# We will use the train split of SQuAD2.0
original_data_file = "train-v2.0.json"

# The data was mirrored in the following bucket
original_data_location = (
    f"s3://sagemaker-example-files-prod-{aws_region}/datasets/text/squad2.0/{original_data_file}"
)
S3Downloader.download(original_data_location, ".")

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

# In[16]:


template = {
    "prompt": "Ask a question which is related to the following text, but cannot be answered based on the text. Text: {context}",
    "completion": "{question}",
}

with open("template.json", "w") as f:
    json.dump(template, f)

# Next, we are going to reformat the SQuAD 2.0 dataset. The processed data is saved as `task-data.jsonl` file. Given the prompt template defined in above cell, each entry in the `task-data.jsonl` file include **`context`** and **`question`** fields. For demonstration purpose, we limit the number of training examples to be 2000.

# In[17]:


local_data_file = "task-data.jsonl"  # any name with .jsonl extension

with open(original_data_file) as f:
    data = json.load(f)

def preprocess_data(local_data_file, data, num_maximum_example):
    num_example_idx = 0  
    with open(local_data_file, "w") as f:
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                # iterate over questions for a given paragraph
                for qas in paragraph["qas"]:
                    if qas["is_impossible"]:
                        # the question is relevant, but cannot be answered
                        example = {"context": paragraph["context"], "question": qas["question"]}
                        json.dump(example, f)
                        f.write("\n")
                        num_example_idx += 1
                        if num_example_idx >= num_maximum_example:
                            return

preprocess_data(local_data_file=local_data_file, data=data, num_maximum_example=10000)

# Upload the prompt template (`template.json`) and training data (`task-data.jsonl`) into S3 bucket.

# In[18]:


from sagemaker.s3 import S3Uploader

training_dataset_s3_path = f"s3://{output_bucket}/train_data"
S3Uploader.upload(local_data_file, training_dataset_s3_path)
S3Uploader.upload("template.json", training_dataset_s3_path)
print(f"{bold}training data:{unbold} {training_dataset_s3_path}")

# ### 2.2. Prepare training parameters

# In[19]:


from sagemaker import hyperparameters

my_hyperparameters = hyperparameters.retrieve_default(model_id=model_id, model_version=model_version)
print(my_hyperparameters)

# Overwrite the hyperparameters

# In[20]:


my_hyperparameters["epoch"] = "2"
my_hyperparameters["per_device_train_batch_size"] = "2"
my_hyperparameters["gradient_accumulation_steps"] = "2"
my_hyperparameters["instruction_tuned"] = "True"
print(my_hyperparameters)

# Validate hyperparameters

# In[21]:


hyperparameters.validate(model_id=model_id, model_version=model_version, hyperparameters=my_hyperparameters)

# ### 2.3. Starting training

# Note. The parameter `load_best_model_at_end` (Whether or not to load the best model found during training at the end of training. When this option is enabled, the best checkpoint will always be saved) is set as "True" by default. During loading the best model checkpoints at the end of training (HuggingFace will load the best model checkpoints before saving it), there is overhead of memory usage which can lead to Out-Of-Memory error.
# 
# If setting `load_best_model_at_end`, we recommend to use `ml.g5.48xlarge`; if not, we recommend to use `ml.g5.12xlarge`.

# In[22]:


from sagemaker.jumpstart.estimator import JumpStartEstimator

instruction_tuned_estimator = JumpStartEstimator(
    model_id=model_id,
    hyperparameters=my_hyperparameters,
    instance_type="ml.g5.48xlarge",
)
instruction_tuned_estimator.fit(
    {"train": training_dataset_s3_path}, logs=True
)

# Extract Training performance metrics. Performance metrics such as training loss and validation accuracy/loss can be accessed through cloudwatch while the training. We can also fetch these metrics and analyze them within the notebook.

# In[ ]:


from sagemaker import TrainingJobAnalytics

training_job_name = instruction_tuned_estimator.latest_training_job.job_name

df = TrainingJobAnalytics(training_job_name=training_job_name).dataframe()
df.head(10)

# ### 2.4. Deploying inference endpoints

# In[ ]:


instruction_tuned_predictor = instruction_tuned_estimator.deploy()

# ### 2.5. Running inference queries and compare model performances
# 
# We examine three examples as listed in variable `test_paragraphs`. The prompt as defined in variable `prompt` asks the model to ask a question based on the context and make sure the question **cannot** be answered from the context. 
# 
# We compare the performance of pre-trained Falcon instruct 7b (`huggingface-llm-falcon-7b-instruct-bf16`) that we deployed in [Section 1](#1.-Deploying-Falcon-model-for-inference) and fine-tuned Falcon instruct 7b.

# In[ ]:


prompt = "Ask a question which is related to the following text, but cannot be answered based on the text. Text: {context}"

# Sources: Wikipedia, AWS Documentation
test_paragraphs = [
    """
Adelaide is the capital city of South Australia, the state's largest city and the fifth-most populous city in Australia. "Adelaide" may refer to either Greater Adelaide (including the Adelaide Hills) or the Adelaide city centre. The demonym Adelaidean is used to denote the city and the residents of Adelaide. The Traditional Owners of the Adelaide region are the Kaurna people. The area of the city centre and surrounding parklands is called Tarndanya in the Kaurna language.
Adelaide is situated on the Adelaide Plains north of the Fleurieu Peninsula, between the Gulf St Vincent in the west and the Mount Lofty Ranges in the east. Its metropolitan area extends 20 km (12 mi) from the coast to the foothills of the Mount Lofty Ranges, and stretches 96 km (60 mi) from Gawler in the north to Sellicks Beach in the south.
""",
    """
Amazon Elastic Block Store (Amazon EBS) provides block level storage volumes for use with EC2 instances. EBS volumes behave like raw, unformatted block devices. You can mount these volumes as devices on your instances. EBS volumes that are attached to an instance are exposed as storage volumes that persist independently from the life of the instance. You can create a file system on top of these volumes, or use them in any way you would use a block device (such as a hard drive). You can dynamically change the configuration of a volume attached to an instance.
We recommend Amazon EBS for data that must be quickly accessible and requires long-term persistence. EBS volumes are particularly well-suited for use as the primary storage for file systems, databases, or for any applications that require fine granular updates and access to raw, unformatted, block-level storage. Amazon EBS is well suited to both database-style applications that rely on random reads and writes, and to throughput-intensive applications that perform long, continuous reads and writes.
""",
    """
Amazon Comprehend uses natural language processing (NLP) to extract insights about the content of documents. It develops insights by recognizing the entities, key phrases, language, sentiments, and other common elements in a document. Use Amazon Comprehend to create new products based on understanding the structure of documents. For example, using Amazon Comprehend you can search social networking feeds for mentions of products or scan an entire document repository for key phrases. 
You can access Amazon Comprehend document analysis capabilities using the Amazon Comprehend console or using the Amazon Comprehend APIs. You can run real-time analysis for small workloads or you can start asynchronous analysis jobs for large document sets. You can use the pre-trained models that Amazon Comprehend provides, or you can train your own custom models for classification and entity recognition. 
All of the Amazon Comprehend features accept UTF-8 text documents as the input. In addition, custom classification and custom entity recognition accept image files, PDF files, and Word files as input. 
Amazon Comprehend can examine and analyze documents in a variety of languages, depending on the specific feature. For more information, see Languages supported in Amazon Comprehend. Amazon Comprehend's Dominant language capability can examine documents and determine the dominant language for a far wider selection of languages.
""",
]

# In[ ]:


parameters = {
    "max_new_tokens": 50,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.8,
    "do_sample": True,
    "temperature": 0.01,    
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

def generate_question(endpoint_name, text):
    expanded_prompt = prompt.replace("{context}", text)
    payload = {"inputs": expanded_prompt, "parameters": parameters}
    query_response = query_endpoint_with_json_payload(json.dumps(payload).encode("utf-8"), endpoint_name=endpoint_name)
    generated_texts = parse_response(query_response)
    print(f"Response: {generated_texts}{newline}")

# In[ ]:


print(f"{bold}Prompt:{unbold} {repr(prompt)}")
for paragraph in test_paragraphs:
    print("-" * 80)
    print(paragraph)
    print("-" * 80)
    print(f"{bold}pre-trained{unbold}")
    generate_question(predictor.endpoint_name, paragraph)
    print(f"{bold}fine-tuned{unbold}")
    generate_question(instruction_tuned_predictor.endpoint_name, paragraph)

# The pre-trained model was not specifically trained to generate unanswerable questions. Despite the input prompt, it tends to generate questions that can be answered from the text. The fine-tuned model is generally better at this task, and the improvement is more prominent for larger models 

# ### 2.6. Clean up the endpoint

# In[ ]:


# Delete the SageMaker endpoint
predictor.delete_model()
predictor.delete_endpoint()
instruction_tuned_predictor.delete_model()
instruction_tuned_predictor.delete_endpoint()

# ## 3. Domain adaptation fine-tuning
# 
# We also have domain adaptation fine-tuning enabled for Falcon models. Different from instruction fine-tuning, you do not need prepare instruction-formatted dataset and can directly use unstructured text document which is demonstrated as below. However, the model that is domain-adaptation fine-tuned may not give concise responses as the instruction-tuned model because of less restrictive requirements on training data formats.

# In this demonstration, we use falcon text generation model `huggingface-llm-falcon-7b-bf16`. This is not an instruction-tuned version of Falcon 7B model. You can also conduct domain adaptation finetuning on top of instruction-tuned model like `huggingface-llm-falcon-7b-instruct-bf16`. However, we generally do not recommend that. 

# In[ ]:


model_id = "huggingface-llm-falcon-7b-bf16"

# We will use financial text from SEC filings to fine tune `huggingface-llm-falcon-7b-bf16` for financial applications. 
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

# ### 3.1. Preparing training data
# 
# The training data of SEC filing of Amazon has been pre-saved in the S3 bucket.

# In[ ]:


from sagemaker.jumpstart.utils import get_jumpstart_content_bucket

# Sample training data is available in this bucket
data_bucket = get_jumpstart_content_bucket(aws_region)
data_prefix = "training-datasets/sec_data"

training_dataset_s3_path = f"s3://{data_bucket}/{data_prefix}/train/"
validation_dataset_s3_path = f"s3://{data_bucket}/{data_prefix}/validation/"

# ### 3.2. Prepare training parameters

# In[ ]:


from sagemaker import hyperparameters

my_hyperparameters = hyperparameters.retrieve_default(model_id=model_id, model_version=model_version)

my_hyperparameters["epoch"] = "3"
my_hyperparameters["per_device_train_batch_size"] = "2"
my_hyperparameters["instruction_tuned"] = "False"
print(my_hyperparameters)

# Validate hyperparameters

# In[ ]:


hyperparameters.validate(model_id=model_id, model_version=model_version, hyperparameters=my_hyperparameters)

# ### 3.3. Starting training

# In[ ]:


from sagemaker.jumpstart.estimator import JumpStartEstimator

domain_adaptation_estimator = JumpStartEstimator(
    model_id=model_id,
    hyperparameters=my_hyperparameters,
    instance_type="ml.p3dn.24xlarge",
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

# ### 3.4. Deploying inference endpoints
# 
# We deploy the domain-adaptation fine-tuned and pretrained models separately, and compare their performances.
# 
# We firstly deploy the domain-adaptation fine-tuned model.

# In[ ]:


domain_adaptation_predictor = domain_adaptation_estimator.deploy()

# Next, we deploy the pre-trained `huggingface-llm-falcon-7b-bf16`. 

# In[ ]:


my_model = JumpStartModel(model_id=model_id)
pretrained_predictor = my_model.deploy()

# ### 3.5. Running inference queries and compare model performances

# In[ ]:


parameters = {
    "max_new_tokens": 300,
    "top_k": 50,
    "top_p": 0.8,
    "do_sample": True,
    "temperature": 1,
}

def generate_response(endpoint_name, text):
    payload = {"inputs": f"{text}:", "parameters": parameters}
    query_response = query_endpoint_with_json_payload(json.dumps(payload).encode("utf-8"), endpoint_name=endpoint_name)
    generated_texts = parse_response(query_response)
    print(f"Response: {generated_texts}{newline}")

# In[ ]:


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

# ### 3.6. Clean up the endpoint

# In[ ]:


# # Delete the SageMaker endpoint
# pretrained_predictor.delete_model()
# pretrained_predictor.delete_endpoint()
# domain_adaptation_predictor.delete_model()
# domain_adaptation_predictor.delete_endpoint()
