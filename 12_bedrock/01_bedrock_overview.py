#!/usr/bin/env python
# coding: utf-8

# # Amazon Bedrock Introduction
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 
# ---
# 
# In this demo notebook, we demonstrate how to use the [`boto3` Python SDK](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) to work with [Amazon Bedrock](https://aws.amazon.com/bedrock/) Foundation Models.
# 
# ---

# In[2]:


# %pip install --no-build-isolation --force-reinstall \
#     "boto3>=1.28.57" \
#     "awscli>=1.29.57" \
#     "botocore>=1.31.57"

# This notebook demonstrates invoking Bedrock models directly using the AWS SDK, but for later notebooks in the workshop you'll also need to install [LangChain](https://github.com/hwchase17/langchain):

# In[3]:


%pip install --quiet langchain==0.0.309

# ---
# 
# ## Create the boto3 client
# 
# Interaction with the Bedrock API is done via the AWS SDK for Python: [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html).
# 
# #### Use different clients
# The boto3 provides different clients for Amazon Bedrock to perform different actions. The actions for [`InvokeModel`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html) and [`InvokeModelWithResponseStream`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html) are supported by Amazon Bedrock Runtime where as other operations, such as [ListFoundationModels](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html), are handled via [Amazon Bedrock client](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Amazon_Bedrock.html).
# 
# #### Use the default credential chain
# 
# If you are running this notebook from [Amazon Sagemaker Studio](https://aws.amazon.com/sagemaker/studio/) and your Sagemaker Studio [execution role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) has permissions to access Bedrock you can just run the cells below as-is. This is also the case if you are running these notebooks from a computer whose default AWS credentials have access to Bedrock.
# 
# #### Use a different AWS Region
# 
# If you're running this notebook from your own computer or a SageMaker notebook in a different AWS Region from where Bedrock is set up, you can un-comment the `os.environ['AWS_DEFAULT_REGION']` line below and specify the region to use.
# 
# #### Use a specific profile
# 
# In case you're running this notebook from your own computer where you have setup the AWS CLI with multiple profiles, and the profile which has access to Bedrock is not the default one, you can un-comment the `os.environ['AWS_PROFILE']` line below and specify the profile to use.
# 
# #### Use a different role
# 
# In case you or your company has setup a specific, separate [IAM Role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) to access Bedrock, you can specify it by un-commenting the `os.environ['BEDROCK_ASSUME_ROLE']` line below. Ensure that your current user or role have permissions to [assume](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html) such role.
# 
# #### A note about `langchain`
# 
# The Bedrock classes provided by `langchain` create a Bedrock boto3 client by default. To customize your Bedrock configuration, we recommend to explicitly create the Bedrock client using the method below, and pass it to the [`langchain.Bedrock`](https://python.langchain.com/docs/integrations/llms/bedrock) class instantiation method using `client=bedrock_runtime`

# In[4]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# #### Validate the connection
# 
# We can check the client works by trying out the `list_foundation_models()` method, which will tell us all the models available for us to use 

# In[5]:


bedrock.list_foundation_models()

# ---
# 
# ## `InvokeModel` body and output
# 
# The `invoke_model()` method of the Amazon Bedrock runtime client (`InvokeModel` API) will be the primary method we use for most of our Text Generation and Processing tasks - whichever model we're using.
# 
# Although the method is shared, the format of input and output varies depending on the foundation model used - as described below:

# ### Amazon Titan Large
# 
# #### Input
# ```json
# {   
#     "inputText": "<prompt>",
#     "textGenerationConfig" : { 
#         "maxTokenCount": 512,
#         "stopSequences": [],
#         "temperature": 0.1,  
#         "topP": 0.9
#     }
# }
# ```
# 
# #### Output
# 
# ```json
# {
#     "inputTextTokenCount": 613,
#     "results": [{
#         "tokenCount": 219,
#         "outputText": "<output>"
#     }]
# }
# ```

# ### AI21 Jurassic (Grande and Jumbo) 
# 
# #### Input
# 
# ```json
# {
#     "prompt": "<prompt>",
#     "maxTokens": 200,
#     "temperature": 0.5,
#     "topP": 0.5,
#     "stopSequences": [],
#     "countPenalty": {"scale": 0},
#     "presencePenalty": {"scale": 0},
#     "frequencyPenalty": {"scale": 0}
# }
# ```
# 
# #### Output
# 
# ```json
# {
#     "id": 1234,
#     "prompt": {
#         "text": "<prompt>",
#         "tokens": [
#             {
#                 "generatedToken": {
#                     "token": "\u2581who\u2581is",
#                     "logprob": -12.980147361755371,
#                     "raw_logprob": -12.980147361755371
#                 },
#                 "topTokens": null,
#                 "textRange": {"start": 0, "end": 6}
#             },
#             //...
#         ]
#     },
#     "completions": [
#         {
#             "data": {
#                 "text": "<output>",
#                 "tokens": [
#                     {
#                         "generatedToken": {
#                             "token": "<|newline|>",
#                             "logprob": 0.0,
#                             "raw_logprob": -0.01293118204921484
#                         },
#                         "topTokens": null,
#                         "textRange": {"start": 0, "end": 1}
#                     },
#                     //...
#                 ]
#             },
#             "finishReason": {"reason": "endoftext"}
#         }
#     ]
# }
# ```

# ### Anthropic Claude
# 
# #### Input
# 
# ```json
# {
#     "prompt": "\n\nHuman:<prompt>\n\nAnswer:",
#     "max_tokens_to_sample": 300,
#     "temperature": 0.5,
#     "top_k": 250,
#     "top_p": 1,
#     "stop_sequences": ["\n\nHuman:"]
# }
# ```
# 
# #### Output
# 
# ```json
# {
#     "completion": "<output>",
#     "stop_reason": "stop_sequence"
# }
# ```

# ### Stability AI Stable Diffusion XL
# 
# #### Input
# 
# ```json
# {
#     "text_prompts": [
#         {"text": "this is where you place your input text"}
#     ],
#     "cfg_scale": 10,
#     "seed": 0,
#     "steps": 50
# }
# ```
# 
# #### Output
# 
# ```json
# { 
#     "result": "success", 
#     "artifacts": [
#         {
#             "seed": 123, 
#             "base64": "<image in base64>",
#             "finishReason": "SUCCESS"
#         },
#         //...
#     ]
# }
# ```

# ---
# 
# ## Common inference parameter definitions
# 
# ### Randomness and Diversity
# 
# Foundation models support the following parameters to control randomness and diversity in the 
# response.
# 
# **Temperature** – Large language models use probability to construct the words in a sequence. For any 
# given next word, there is a probability distribution of options for the next word in the sequence. When 
# you set the temperature closer to zero, the model tends to select the higher-probability words. When 
# you set the temperature further away from zero, the model may select a lower-probability word.
# 
# In technical terms, the temperature modulates the probability density function for the next tokens, 
# implementing the temperature sampling technique. This parameter can deepen or flatten the density 
# function curve. A lower value results in a steeper curve with more deterministic responses, and a higher 
# value results in a flatter curve with more random responses.
# 
# **Top K** – Temperature defines the probability distribution of potential words, and Top K defines the cut 
# off where the model no longer selects the words. For example, if K=50, the model selects from 50 of the 
# most probable words that could be next in a given sequence. This reduces the probability that an unusual 
# word gets selected next in a sequence.
# In technical terms, Top K is the number of the highest-probability vocabulary tokens to keep for Top-
# K-filtering - This limits the distribution of probable tokens, so the model chooses one of the highest-
# probability tokens.
# 
# **Top P** – Top P defines a cut off based on the sum of probabilities of the potential choices. If you set Top 
# P below 1.0, the model considers the most probable options and ignores less probable ones. Top P is 
# similar to Top K, but instead of capping the number of choices, it caps choices based on the sum of their 
# probabilities.
# For the example prompt "I hear the hoof beats of ," you may want the model to provide "horses," 
# "zebras" or "unicorns" as the next word. If you set the temperature to its maximum, without capping 
# Top K or Top P, you increase the probability of getting unusual results such as "unicorns." If you set the 
# temperature to 0, you increase the probability of "horses." If you set a high temperature and set Top K or 
# Top P to the maximum, you increase the probability of "horses" or "zebras," and decrease the probability 
# of "unicorns."
# 
# ### Length
# 
# The following parameters control the length of the generated response.
# 
# **Response length** – Configures the minimum and maximum number of tokens to use in the generated 
# response.
# 
# **Length penalty** – Length penalty optimizes the model to be more concise in its output by penalizing 
# longer responses. Length penalty differs from response length as the response length is a hard cut off for 
# the minimum or maximum response length.
# 
# In technical terms, the length penalty penalizes the model exponentially for lengthy responses. 0.0 
# means no penalty. Set a value less than 0.0 for the model to generate longer sequences, or set a value 
# greater than 0.0 for the model to produce shorter sequences.
# 
# ### Repetitions
# 
# The following parameters help control repetition in the generated response.
# 
# **Repetition penalty (presence penalty)** – Prevents repetitions of the same words (tokens) in responses. 
# 1.0 means no penalty. Greater than 1.0 decreases repetition.

# ---
# 
# ## Try out the models
# 
# With some theory out of the way, let's see the models in action! Run the cells below to see basic, synchronous example invocations for each model:

# In[6]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# ### Amazon Titan Express

# In[7]:


# If you'd like to try your own prompt, edit this parameter!
prompt_data = """Command: Write me a blog about making strong business decisions as a leader.

Blog:
"""

# In[8]:


try:
    
    body = json.dumps({"inputText": prompt_data})
    modelId = "amazon.titan-text-express-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print(response_body.get("results")[0].get("outputText"))

except botocore.exceptions.ClientError as error:
    
    if error.response['Error']['Code'] == 'AccessDeniedException':
           print(f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        
    else:
        raise error

# ### Anthropic Claude

# In[9]:


# If you'd like to try your own prompt, edit this parameter!
prompt_data = """Human: Write me a blog about making strong business decisions as a leader.

Assistant:
"""

# In[10]:


body = json.dumps({"prompt": prompt_data, "max_tokens_to_sample": 500})
modelId = "anthropic.claude-v2"  # change this to use a different version from the model provider
accept = "application/json"
contentType = "application/json"

try:
    
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print(response_body.get("completion"))

except botocore.exceptions.ClientError as error:
    
    if error.response['Error']['Code'] == 'AccessDeniedException':
           print(f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        
    else:
        raise error

# ### AI21 Jurassic

# In[11]:


body = json.dumps({"prompt": prompt_data, "maxTokens": 200})
modelId = "ai21.j2-mid-v1"  # change this to use a different version from the model provider
accept = "application/json"
contentType = "application/json"

try:
    
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print(response_body.get("completions")[0].get("data").get("text"))

except botocore.exceptions.ClientError as error:
    
    if error.response['Error']['Code'] == 'AccessDeniedException':
           print(f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        
    else:
        raise error

# ### Stability Stable Diffusion XL

# In[12]:


prompt_data = "a fine image of an astronaut riding a horse on Mars"
body = json.dumps({
    "text_prompts": [{"text": prompt_data}],
    "cfg_scale": 10,
    "seed": 20,
    "steps": 50
})
modelId = "stability.stable-diffusion-xl"
accept = "application/json"
contentType = "application/json"

try:
    
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print(response_body["result"])
    print(f'{response_body.get("artifacts")[0].get("base64")[0:80]}...')

except botocore.exceptions.ClientError as error:
    
    if error.response['Error']['Code'] == 'AccessDeniedException':
           print(f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        
    else:
        raise error

# **Note:** The output is a [base64 encoded](https://docs.python.org/3/library/base64.html) string of the image data. You can use any image processing library (such as [Pillow](https://pillow.readthedocs.io/en/stable/)) to decode the image as in the example below:
# 
# ```python
# import base64
# import io
# from PIL import Image
# 
# base_64_img_str = response_body.get("artifacts")[0].get("base64")
# image = Image.open(io.BytesIO(base64.decodebytes(bytes(base_64_img_str, "utf-8"))))
# ```

# ## Generate streaming output
# 
# For large language models, it can take noticeable time to generate long output sequences. Rather than waiting for the entire response to be available, latency-sensitive applications may like to **stream** the response to users.
# 
# Run the code below to see how you can achieve this with Bedrock's `invoke_model_with_response_stream()` method - returning the response body in separate chunks.

# In[13]:


from IPython.display import clear_output, display, display_markdown, Markdown

body = json.dumps({"inputText": prompt_data})
modelId = "amazon.titan-text-express-v1"  # (Change this, and the request body, to try different models)
accept = "application/json"
contentType = "application/json"

try:
    
    response = bedrock_runtime.invoke_model_with_response_stream(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    stream = response.get('body')
    output = []

    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                text = chunk_obj['outputText']
                clear_output(wait=True)
                output.append(text)
                display_markdown(Markdown(''.join(output)))
            
except botocore.exceptions.ClientError as error:
    
    if error.response['Error']['Code'] == 'AccessDeniedException':
           print(f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        
    else:
        raise error

# ### Anthropic Claude

# In[14]:


# If you'd like to try your own prompt, edit this parameter!
prompt_data = """Human: Write me a blog about making strong business decisions as a leader.

Assistant:
"""

# In[15]:


import botocore
from IPython.display import clear_output, display, display_markdown, Markdown

body = json.dumps({"prompt": prompt_data, "max_tokens_to_sample": 500})
modelId = "anthropic.claude-instant-v1"  # (Change this to try different model versions)
accept = "application/json"
contentType = "application/json"

try:
    
    response = bedrock_runtime.invoke_model_with_response_stream(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    stream = response.get('body')
    output = []

    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                text = chunk_obj['completion']
                clear_output(wait=True)
                output.append(text)
                display_markdown(Markdown(''.join(output)))
            
except botocore.exceptions.ClientError as error:
    
    if error.response['Error']['Code'] == 'AccessDeniedException':
           print(f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        
    else:
        raise error

# ## Generate embeddings
# 
# Use text embeddings to convert text into meaningful vector representations. You input a body of text 
# and the output is a (1 x n) vector. You can use embedding vectors for a wide variety of applications. 
# Bedrock currently offers Titan Embeddings for text embedding that supports text similarity (finding the 
# semantic similarity between bodies of text) and text retrieval (such as search).
# 
# At the time of writing you can use `amazon.titan-embed-g1-text-02` as embedding model via the API. The input text size is 8192 tokens and the output vector length is 1536.
# 
# To use a text embeddings model, use the InvokeModel API operation or the Python SDK.
# Use InvokeModel to retrieve the vector representation of the input text from the specified model.
# 
# 
# 
# #### Input
# 
# ```json
# {
#     "inputText": "<text>"
# }
# ```
# 
# #### Output
# 
# ```json
# {
#     "embedding": []
# }
# ```
# 

# Let's see how to generate embeddings of some text:

# In[16]:


prompt_data = "Amazon Bedrock supports foundation models from industry-leading providers such as \
AI21 Labs, Anthropic, Stability AI, and Amazon. Choose the model that is best suited to achieving \
your unique goals."

# In[17]:


body = json.dumps({"inputText": prompt_data})
modelId = "amazon.titan-embed-text-v1"  # (Change this to try different embedding models)
accept = "application/json"
contentType = "application/json"

try:
    
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    embedding = response_body.get("embedding")
    print(f"The embedding vector has {len(embedding)} values\n{embedding[0:3]+['...']+embedding[-3:]}")

except botocore.exceptions.ClientError as error:
    
    if error.response['Error']['Code'] == 'AccessDeniedException':
           print(f"\x1b[41m{error.response['Error']['Message']}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
        
    else:
        raise error

# ## Next steps
# 
# In this notebook we showed some basic examples of invoking Amazon Bedrock models using the AWS Python SDK. You're now ready to explore the other labs to dive deeper on different use-cases and patterns.

# In[ ]:



