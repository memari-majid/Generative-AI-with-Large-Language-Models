#!/usr/bin/env python
# coding: utf-8

# # Abstractive Text Summarization with Amazon Titan
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 
# 
# # Text summarization with small files with Amazon Titan
# 

# ## Overview
# When we work with large documents, we can face some challenges as the input text might not fit into the model context length, or the model hallucinates with large documents, or, out of memory errors, etc.
# 
# To solve those problems, we are going to show an architecture that is based on the concept of chunking and chaining prompts. This architecture is leveraging [LangChain](https://python.langchain.com/docs/get_started/introduction.html) which is a popular framework for developing applications powered by language models.
# 
# ### Architecture
# 
# ![](./images/42-text-summarization-2.png)
# 
# In this architecture:
# 
# 1. A large document (or a giant file appending small ones) is loaded
# 1. Langchain utility is used to split it into multiple smaller chunks (chunking)
# 1. First chunk is sent to the model; Model returns the corresponding summary
# 1. Langchain gets next chunk and appends it to the returned summary and sends the combined text as a new request to the model; the process repeats until all chunks are processed
# 1. In the end, you have final summary based on entire content
# 
# ### Use case
# This approach can be used to summarize call transcripts, meetings transcripts, books, articles, blog posts, and other relevant content.
# 
# ## Setup

# In[2]:


# Make sure you run `download-dependencies.sh` from the root of the repository to download the dependencies before running this cell
# %pip install -U boto3 botocore --force-reinstall --quiet
# %pip install -U transformers==4.34.0 --quiet
# %pip install -U langchain==0.0.309 --quiet

# In[3]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# ## Summarize long text 
# 
# ### Configuring LangChain with Boto3
# 
# LangChain allows you to access Bedrock once you pass boto3 session information to LangChain. If you pass None as the boto3 session information to LangChain, LangChain tries to get session information from your environment.
# In order to ensure the right client is used we are going to instantiate one thanks to a utility method.
# 
# You need to specify LLM for LangChain Bedrock class, and can pass arguments for inference. Here you specify Amazon Titan Text Large in `model_id` and pass Titan's inference parameter in `textGenerationConfig`.

# In[4]:


from langchain.llms.bedrock import Bedrock

model_kwargs_titan = { 
    "maxTokenCount": 4096,
    "stopSequences": [],
    "temperature": 0,  
    "topP": 1
}

llm = Bedrock(
    model_id="amazon.titan-text-express-v1", 
    client=bedrock_runtime, 
    model_kwargs=model_kwargs_titan
)

# ### Loading a text file with many tokens
# 
# In `letters` directory, you can find a text file of [Amazon's CEO letter to shareholders in 2022](https://www.aboutamazon.com/news/company-news/amazon-ceo-andy-jassy-2022-letter-to-shareholders). The following cell loads the text file and counts the number of tokens in the file. 
# 
# You will see warning indicating the number of tokens in the text file exceeeds the maximum number of tokens fot his model.

# In[5]:


shareholder_letter = "./letters/2022-letter.txt"

with open(shareholder_letter, "r") as file:
    letter = file.read()
    
llm.get_num_tokens(letter)

# ### Splitting the long text into chunks
# 
# The text is too long to fit in the prompt, so we will split it into smaller chunks.
# `RecursiveCharacterTextSplitter` in LangChain supports splitting long text into chunks recursively until size of each chunk becomes smaller than `chunk_size`. A text is separated with `separators=["\n\n", "\n"]` into chunks, which avoids splitting each paragraph into multiple chunks.
# 
# Using 6,000 characters per chunk, we can get summaries for each portion separately. The number of tokens, or word pieces, in a chunk depends on the text.

# In[6]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=100
)

docs = text_splitter.create_documents([letter])

# In[7]:


num_docs = len(docs)

num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)

print(
    f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens"
)

# ### Summarizing chunks and combining them

# Assuming that the number of tokens is consistent in the other docs we should be good to go. Let's use LangChain's [load_summarize_chain](https://python.langchain.com/en/latest/use_cases/summarization.html) to summarize the text. `load_summarize_chain` provides three ways of summarization: `stuff`, `map_reduce`, and `refine`. 
# - `stuff` puts all the chunks into one prompt. Thus, this would hit the maximum limit of tokens.
# - `map_reduce` summarizes each chunk, combines the summary, and summarizes the combined summary. If the combined summary is too large, it would raise error.
# - `refine` summarizes the first chunk, and then summarizes the second chunk with the first summary. The same process repeats until all chunks are summarized.
# 
# `map_reduce` and `refine` invoke LLM multiple times and takes time for obtaining final summary. 
# Let's try `map_reduce` here. 

# In[8]:


# Set verbose=True if you want to see the prompts being used
from langchain.chains.summarize import load_summarize_chain
summary_chain = load_summarize_chain(llm=llm, 
                                     chain_type="map_reduce", 
                                     verbose=True)

# In[ ]:


output = summary_chain.run(docs)

# In[ ]:


print(output.strip())

# In[ ]:



