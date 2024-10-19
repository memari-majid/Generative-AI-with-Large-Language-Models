#!/usr/bin/env python
# coding: utf-8

# # Text summarization with small files with Amazon Titan
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 

# ## Overview
# 
# In this example, you are going to ingest a small amount of data (String data) directly into Amazon Bedrock API (using Amazon Titan model) and give it an instruction to summarize the respective text.
# 
# ### Architecture
# 
# ![](./images/41-text-simple-1.png)
# 
# In this architecture:
# 
# 1. A small piece of text (or small file) is loaded
# 1. A foundational model processes those data
# 1. Model returns a response with the summary of the ingested text
# 
# ### Use case
# 
# This approach can be used to summarize call transcripts, meetings transcripts, books, articles, blog posts, and other relevant content.
# 
# ### Challenges
# This approach can be used when the input text or file fits within the model context length. In notebook `02.long-text-summarization-titan.ipynb`, we will explore an approach to address the challenge when users have large document(s) that exceed the token limit.
# 
# 
# ## Setup

# In[2]:


# %pip install -U boto3 botocore --force-reinstall --quiet

# #### Now let's set up our connection to the Amazon Bedrock SDK using Boto3

# In[3]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# ## Summarizing a short text with boto3
#  
# To learn detail of API request to Amazon Bedrock, this notebook introduces how to create API request and send the request via Boto3 rather than relying on langchain, which gives simpler API by wrapping Boto3 operation. 

# ### Request Syntax of InvokeModel in Boto3
# 
# 
# We use `InvokeModel` API for sending request to a foundation model. Here is an example of API request for sending text to Amazon Titan Text Large. Inference parameters in `textGenerationConfig` depends on the model that you are about to use. Inference paramerters of Amazon Titan Text are:
# - **maxTokenCount** configures the max number of tokens to use in the generated response. (int, defaults to 512)
# - **stopSequences** is used to make the model stop at a desired point, such as the end of a sentence or a list. The returned response will not contain the stop sequence.
# - **temperature** modulates the probability density function for the next tokens, implementing the temperature sampling technique. This parameter can be used to deepen or flatten the density function curve. A lower value results in a steeper curve and more deterministic responses, whereas a higher value results in a flatter curve and more random responses. (float, defaults to 0, max value is 1.5)
# - **topP** controls token choices, based on the probability of the potential choices. If you set Top P below 1.0, the model considers only the most probable options and ignores less probable options. The result is more stable and repetitive completions.
# 
# ```python
# response = bedrock_runtime.invoke_model(body={
#                                    "inputText": "this is where you place your input text",
#                                    "textGenerationConfig": {
#                                        "maxTokenCount": 4096,
#                                        "stopSequences": [],
#                                        "temperature":0,
#                                        "topP":1
#                                        },
#                                 },
#                                 modelId="amazon.titan-text-express-v1", 
#                                 accept=accept, 
#                                 contentType=contentType)
# 
# ```
# 
# ### Writing prompt with text to be summarized
# 
# In this notebook, you can use any short text whose tokens are less than the maximum token of a foundation model. As an exmple of short text, let's take one paragraph of an [AWS blog post](https://aws.amazon.com/jp/blogs/machine-learning/announcing-new-tools-for-building-with-generative-ai-on-aws/) about announcement of Amazon Bedrock.
# 
# The prompt starts with an instruction `Please provide a summary of the following text.`, and includes text surrounded by  `<text>` tag. 

# In[4]:


prompt = """
Please provide a summary of the following text. Do not add any information that is not mentioned in the text below.

<text>
AWS took all of that feedback from customers, and today we are excited to announce Amazon Bedrock, \
a new service that makes FMs from AI21 Labs, Anthropic, Stability AI, and Amazon accessible via an API. \
Bedrock is the easiest way for customers to build and scale generative AI-based applications using FMs, \
democratizing access for all builders. Bedrock will offer the ability to access a range of powerful FMs \
for text and images—including Amazons Titan FMs, which consist of two new LLMs we’re also announcing \
today—through a scalable, reliable, and secure AWS managed service. With Bedrock’s serverless experience, \
customers can easily find the right model for what they’re trying to get done, get started quickly, privately \
customize FMs with their own data, and easily integrate and deploy them into their applications using the AWS \
tools and capabilities they are familiar with, without having to manage any infrastructure (including integrations \
with Amazon SageMaker ML features like Experiments to test different models and Pipelines to manage their FMs at scale).
</text>

"""

# ## Creating request body with prompt and inference parameters 
# 
# Following the request syntax of `invoke_model`, you create request body with the above prompt and inference parameters.

# In[5]:


body = json.dumps({"inputText": prompt, 
                   "textGenerationConfig":{
                       "maxTokenCount":4096,
                       "stopSequences":[],
                       "temperature":0,
                       "topP":1
                   },
                  }) 

# ## Invoke foundation model via Boto3
# 
# Here sends the API request to Amazon Bedrock with specifying request parameters `modelId`, `accept`, and `contentType`. Following the prompt, the foundation model in Amazon Bedrock sumamrizes the text.

# In[8]:


modelId = 'amazon.titan-text-express-v1' # change this to use a different version from the model provider
accept = 'application/json'
contentType = 'application/json'

response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
response_body = json.loads(response.get('body').read())

print(response_body.get('results')[0].get('outputText'))

# In the above the Bedrock service generates the entire summary for the given prompt in a single output, this can be slow if the output contains large amount of tokens. 
# 
# Below we explore the option how we can use Bedrock to stream the output such that the user could start consuming it as it is being generated by the model. For this Bedrock supports `invoke_model_with_response_stream` API providing `ResponseStream` that streams the output in form of chunks.

# In[9]:


response = bedrock_runtime.invoke_model_with_response_stream(body=body, modelId=modelId, accept=accept, contentType=contentType)
stream = response.get('body')
output = list(stream)
output

# Instead of generating the entire output, Bedrock sends smaller chunks from the model. This can be displayed in a consumable manner as well.

# In[10]:


from IPython.display import display_markdown, Markdown, clear_output

# In[11]:


response = bedrock_runtime.invoke_model_with_response_stream(body=body, modelId=modelId, accept=accept, contentType=contentType)
stream = response.get('body')
output = []
i = 1
if stream:
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['outputText']
            clear_output(wait=True)
            output.append(text)
            display_markdown(Markdown(''.join(output)))
            i+=1

clear_output(wait=True)
print(''.join(output))

# ## Conclusion
# You have now experimented with using `boto3` SDK which provides a vanilla exposure to Amazon Bedrock API. Using this API you have seen the use case of generating a summary of AWS news about Amazon Bedrock in 2 different ways: entire output and streaming output generation.
# 
# ### Take aways
# - Adapt this notebook to experiment with different models available through Amazon Bedrock such as Anthropic Claude and AI21 Labs Jurassic models.
# - Change the prompts to your specific usecase and evaluate the output of different models.
# - Play with the token length to understand the latency and responsiveness of the service.
# - Apply different prompt engineering principles to get better outputs.
# 
# ## Thank You

# In[ ]:



