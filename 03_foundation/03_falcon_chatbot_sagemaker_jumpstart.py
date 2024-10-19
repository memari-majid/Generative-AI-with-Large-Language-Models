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

# In[ ]:


# !pip install sagemaker --quiet --upgrade --force-reinstall --quiet
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

# ### 2.6. Clean up the endpoint

# In[ ]:


# Delete the SageMaker endpoint
predictor.delete_model()
predictor.delete_endpoint()
