#!/usr/bin/env python
# coding: utf-8

# # Entity Extraction with Claude
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 
# ### Context
# Entity extraction is an NLP technique that allows us to automatically extract specific data from naturally written text, such as news, emails, books, etc.
# That data can then later be saved to a database, used for lookup or any other type of processing.
# 
# Classic entity extraction programs usually limit you to pre-defined classes, such as name, address, price, etc. or require you to provide many examples of types of entities you are interested in.
# By using a LLM for entity extraction in most cases you are only required to specify what you need to extract in natural language. This gives you flexibility and accuracy in your queries while saving time by removing necessity of data labeling.
# 
# In addition, LLM entity extraction can be used to help you assemble a dataset to later creat a customised solution for your use case, such as [Amazon Comprehend custom entity](https://docs.aws.amazon.com/comprehend/latest/dg/custom-entity-recognition.html) recognition.

# ## Setup
# 
# Before running the rest of this notebook, you'll need to run the cells below to (ensure necessary libraries are installed and) connect to Bedrock.
# 
# In this notebook, we'll also need some extra dependencies:
# 
# - [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/), to easily extract data from XML tags in Claude prompts and outputs.

# In[2]:


# %pip install --no-build-isolation --force-reinstall \
#     "boto3>=1.28.57" \
#     "awscli>=1.29.57" \
#     "botocore>=1.31.57" \
#     langchain==0.0.309 beautifulsoup4

# In[3]:


import warnings
warnings.filterwarnings('ignore')

# In[4]:


#### Un comment the following lines to run from your local environment outside of the AWS account with Bedrock access

#import os
#os.environ['BEDROCK_ASSUME_ROLE'] = '<YOUR_VALUES>'
#os.environ['AWS_PROFILE'] = '<YOUR_VALUES>'

# In[5]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# ## Configure langchain
# 
# We begin with instantiating the LLM. Here we are using Anthropic Claude v2 for text generation.
# 
# Note: It is possible to choose other models available with Bedrock. You can replace the `model_id` as follows to change the model.
# 
# `llm = Bedrock(model_id="amazon.titan-tg1-large")`

# In[6]:


from langchain.llms.bedrock import Bedrock

# - create the Anthropic Model
llm = Bedrock(
    model_id="anthropic.claude-v2",
    client=bedrock_runtime,
    model_kwargs={
        "max_tokens_to_sample": 200,
        "temperature": 0, # Using 0 to get reproducible results
        "stop_sequences": ["\n\nHuman:"]
    }
)

# ## Entity Extraction
# Now that we have our LLM initialised, we can start extracting entities.
# 
# For this exercise we will pretend to be an online bookstore that receives questions and orders by email.
# Our task would be to extract relevant information from the email to process the order.
# 
# Let's begin by taking a look at the sample email:

# In[7]:


from pathlib import Path

emails_dir = Path(".") / "emails"
with open(emails_dir / "00_treasure_island.txt") as f:
    book_question_email = f.read()

print(book_question_email)

# ### Basic approach
# 
# For basic cases we can directly ask the model to return the result.
# Let's try extracting the name of the book.

# In[8]:


query = f"""

Human: Given the email inside triple-backticks, please read it and analyse the contents.
If a name of a book is mentioned, return it, otherwise return nothing.

Email: ```
{book_question_email}
```

Assistant:"""

# In[9]:


result = llm(query)
print(result.strip())

# ### Model specific prompts
# 
# While basic approach works, to achieve best results we recommend to customise your prompts for the particular model you will be using.
# In this example we are using `anthropic.claude-v2`, [prompt guide for which can be found here](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design).
# 
# Here is the a more optimised prompt for Claude v2.

# In[10]:


prompt = """

Human: Given the email provided, please read it and analyse the contents.
If a name of a book is mentioned, return it.
If no name is mentioned, return empty string.
The email will be given between <email></email> XML tags.

<email>
{email}
</email>

Return the name of the book between <book></book> XML tags.

Assistant:"""

# In[11]:


query = prompt.format(email=book_question_email)
result = llm(query)
print(result.strip())

# To extract results easier, we can use a helper function:

# In[12]:


from bs4 import BeautifulSoup

def extract_by_tag(response: str, tag: str, extract_all=False) -> str | list[str] | None:
    soup = BeautifulSoup(response)
    results = soup.find_all(tag)
    if not results:
        return
        
    texts = [res.get_text() for res in results]
    if extract_all:
        return texts
    return texts[-1]

# In[13]:


extract_by_tag(result, "book")

# We can check that our model doesn't return arbitrary results when no appropriate information is given (also know as 'hallucination'), by running our prompt on other emails.

# In[14]:


with open(emails_dir / "01_return.txt") as f:
    return_email = f.read()

print(return_email)

# In[15]:


query = prompt.format(email=return_email)
result = llm(query)
print(result.strip())

# Using tags also allows us to extract multiple pieces of information at the same time and makes extraction much easier.
# In the following prompt we will extract not just the book name, but any questions, requests and customer name.

# In[16]:


prompt = """

Human: Given email provided , please read it and analyse the contents.

Please extract the following information from the email:
- Any questions the customer is asking, return it inside <questions></questions> XML tags.
- The customer full name, return it inside <name></name> XML tags.
- Any book names the customer mentions, return it inside <books></books> XML tags.

If a particular bit of information is not present, return an empty string.
Make sure that each question can be understoon by itself, incorporate context if requred.
Each returned question should be concise, remove extra information if possible.
The email will be given between <email></email> XML tags.

<email>
{email}
</email>

Return each question inside <question></question> XML tags.
Return the name of each book inside <book></book> XML tags.

Assistant:"""

# In[17]:


query = prompt.format(email=book_question_email)
result = llm(query)
print(result.strip())

# In[18]:


extract_by_tag(result, "question", extract_all=True)

# In[19]:


extract_by_tag(result, "name")

# In[20]:


extract_by_tag(result, "book", extract_all=True)

# ## Conclusion
# 
# Entity extraction is a powerful technique using which you can extract arbitrary data using plain text descriptions.
# 
# This is particularly useful when you need to extract specific data which doesn't have clear structure. In such cases regex and other traditional extraction techniques can be very difficult to implement.
# 
# ### Take aways
# - Adapt this notebook to experiment with different models available through Amazon Bedrock such as Amazon Titan and AI21 Labs Jurassic models.
# - Change the prompts to your specific usecase and evaluate the output of different models.
# - Apply different prompt engineering principles to get better outputs. Refer to the prompt guide for your chosen model for recommendations, e.g. [here is the prompt guide for Claude](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design).
