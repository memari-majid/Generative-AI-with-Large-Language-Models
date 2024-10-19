#!/usr/bin/env python
# coding: utf-8

# # Maximizing AI Potentials: Leveraging Foundational Models from Amazon Bedrock and Amazon OpenSearch Serverless as Vector Engine
# 
# ### Context
# Amazon Bedrock is a fully managed service that provides access to FMs from third-party providers and Amazon; available via an API. With Bedrock, you can choose from a variety of models to find the one that’s best suited for your use case. On one hand Amazon Bedrock provides an option to generate vectors as well as summarizezation of texts, then on other hands vector engine for Amazon OpenSearch Serverless complements it by providing a machinsm to store those vectors and run semantic search against those vectors. 
# 
# In this sample notebook you will explore some of the most common usage patterns we are seeing with our customers for Generative AI such as generating text and images, creating value for organizations by improving productivity. This is achieved by leveraging foundation models to help in composing emails, summarizing text, answering questions, building chatbots, and creating images.
# 
# ### Challenges
# - How to manage large document(s) that exceed the token limit
# - How to find the document(s) relevant to the question being asked
# 
# ### Proposal
# To the above challenges, this notebook proposes the following strategy
# #### Prepare documents
# ![Embeddings](./images/Embeddings_lang.png)
# 
# Before being able to answer the questions, the documents must be processed and a stored in a document store index
# - Load the documents
# - Process and split them into smaller chunks
# - Create a numerical vector representation of each chunk using Amazon Bedrock Titan Embeddings model
# - Create an index using the chunks and the corresponding embeddings and store into OpenSearch Serverless
# #### Ask question
# ![Question](./images/Chatbot_lang.png)
# 
# When the documents index is prepared, you are ready to ask the questions and relevant documents will be fetched based on the question being asked. Following steps will be executed.
# - Create an embedding of the input question
# - Compare the question embedding with the embeddings stored in OpenSearch Serverless
# - Fetch the (top N) relevant document chunks using vector engine
# - Add those chunks as part of the context in the prompt
# - Send the prompt to the model under Amazon Bedrock
# - Get the contextual answer based on the documents retrieved

# ## Usecase
# #### Dataset
# To explain this architecture pattern we are using the documents from IRS. These documents explain topics such as:
# - Original Issue Discount (OID) Instruments
# - Reporting Cash Payments of Over $10,000 to IRS
# - Employer's Tax Guide
# 
# The model will try to answer from the documents in easy language.
# 

# ## Implementation
# In order to follow the RAG approach this notebook is using the LangChain framework where it has integrations with different services and tools that allow efficient building of patterns such as RAG. We will be using the following tools:
# 
# - **LLM (Large Language Model)**: Anthropic Claude V1 available through Amazon Bedrock
# - **Vector Store**: vector engine for Amazon OpenSearch Serverless
#   In this notebook we are using OpenSearch Serverless as a vector-store to store both the embeddings and the documents. 
# - **Index**: VectorIndex - This model will be used to understand the document chunks and provide an answer in human friendly manner.
# - **Embeddings Model**: Amazon Titan Embeddings available through Amazon Bedrock
# 
#   This model will be used to generate a numerical representation of the textual documents
# - **Document Loader**: PDF Loader available through LangChain
# 
#   This is the loader that can load the documents from a source, in this example we are loading the vector embeddings generated from those file chunks to OpenSearch Serverless. 
# 
#   The index helps to compare the input embedding and the document embeddings to find relevant document
# - **Wrapper**: wraps index, vector store, embeddings model and the LLM to abstract away the logic from the user.
# 
# ### Setup
# To run this notebook you would need to install dependencies such as, [PyPDF](https://pypi.org/project/pypdf/)

# In[4]:


%pip install langchain==0.0.309 --force-reinstall --quiet
%pip install pypdf==3.8.1 --force-reinstall --quiet

# In[5]:


%pip install requests_aws4auth opensearch-py

# In[6]:


#### Un comment the following lines to run from your local environment outside of the AWS account with Bedrock access

# import os
# os.environ['BEDROCK_ASSUME_ROLE'] = '<enter role>'
# os.environ['AWS_PROFILE'] = '<aws-profile>'

# In[7]:


import boto3
import json
import os
import sys

# module_path = ".."
# sys.path.append(os.path.abspath(module_path))
# from utils import bedrock, print_ww

# os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
# boto3_bedrock = bedrock.get_bedrock_client(os.environ.get('BEDROCK_ASSUME_ROLE', None))
# print (f"bedrock client {boto3_bedrock}")

# In[54]:


## set up opensearch
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import json

# create open search collection public endpoint
host = '<your opensearch collection endpoint>' # OpenSearch Serverless collection endpoint

region = 'us-west-2'

service = 'aoss'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,
session_token=credentials.token)

# Create an OpenSearch client
client = OpenSearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = awsauth,
    timeout = 300,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

# ### Data Preparation
# Let's first download some of the files to build our document store. For this example we will be using public IRS documents from [here](https://www.irs.gov/publications).

# In[10]:


!mkdir -p ./data

from urllib.request import urlretrieve
urls = [
    'https://s2.q4cdn.com/299287126/files/doc_financials/2023/ar/2022-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2022/ar/2021-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2021/ar/Amazon-2020-Shareholder-Letter-and-1997-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2020/ar/2019-Shareholder-Letter.pdf'
]

filenames = [
    'AMZN-2022-Shareholder-Letter.pdf',
    'AMZN-2021-Shareholder-Letter.pdf',
    'AMZN-2020-Shareholder-Letter.pdf',
    'AMZN-2019-Shareholder-Letter.pdf'
]

metadata = [
    dict(year=2022, source=filenames[0]),
    dict(year=2021, source=filenames[1]),
    dict(year=2020, source=filenames[2]),
    dict(year=2019, source=filenames[3])]

data_root = "./data/"

for idx, url in enumerate(urls):
    file_path = data_root + filenames[idx]
    urlretrieve(url, file_path)

# As part of Amazon's culture, the CEO always includes a copy of the 1997 Letter to Shareholders with every new release. This will cause repetition, take longer to generate embeddings, and may skew your results. In the next section you will take the downloaded data, trim the 1997 letter (last 3 pages) and overwrite them as processed files.

# In[11]:


import glob
from pypdf import PdfReader, PdfWriter

local_pdfs = glob.glob(data_root + '*.pdf')

for local_pdf in local_pdfs:
    pdf_reader = PdfReader(local_pdf)
    pdf_writer = PdfWriter()
    for pagenum in range(len(pdf_reader.pages)-3):
        page = pdf_reader.pages[pagenum]
        pdf_writer.add_page(page)

    with open(local_pdf, 'wb') as new_file:
        new_file.seek(0)
        pdf_writer.write(new_file)
        new_file.truncate()

# After downloading we can load the documents with the help of [DirectoryLoader from PyPDF available under LangChain](https://python.langchain.com/en/latest/reference/modules/document_loaders.html) and splitting them into smaller chunks.
# 
# Note: The retrieved document/text should be large enough to contain enough information to answer a question; but small enough to fit into the LLM prompt. Also the embeddings model has a limit of the length of input tokens limited to 512 tokens, which roughly translates to ~2000 characters. For the sake of this use-case we are creating chunks of roughly 1000 characters with an overlap of 100 characters using [RecursiveCharacterTextSplitter](https://python.langchain.com/en/latest/modules/indexes/text_splitters/examples/recursive_text_splitter.html).

# In[12]:


import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

documents = []

for idx, file in enumerate(filenames):
    loader = PyPDFLoader(data_root + file)
    document = loader.load()
    for document_fragment in document:
        document_fragment.metadata = metadata[idx]
        
    print(f'{len(document)} {document}\n')
    documents += document

# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
)

docs = text_splitter.split_documents(documents)

# Before we are proceeding we are looking into some interesting statistics regarding the document preprocessing we just performed:

# In[13]:


avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
print(f'Average length among {len(documents)} documents loaded is {avg_doc_length(documents)} characters.')
print(f'After the split we have {len(docs)} documents as opposed to the original {len(documents)}.')
print(f'Average length among {len(docs)} documents (after split) is {avg_doc_length(docs)} characters.')

# ## Deploy Model for Embedding

# In the following sections you will need to deploy a set of ML Models, one for Embeddings and a LLM for Language Generation. This example assumes you are working inside of SageMaker studio, so you can deploy them yourself or through SageMaker Jumpstart.
# 
# For these examples, you will use `All MiniLM L6 v2` as the embedding model, and `LLaMa-2-7B-chat` as the LLM for language generation. 
# 
# __Note:__ If you choose other options, you may have to adjust the `transform_input` and `transform_output` functions in future sections for embedding and llm to match the models you've selected.
# 
# Refer to the [SageMaker Jumpstart Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html) for details on how to deploy models via Jumpstart.
# 
# If you already have an embedding endpoint deployed, you can skip the following cell, and modify the `embedding_model_endpoint_name` value to match your endpoint.

# __Note: running the following cell will deploy a SageMaker endpoint. You will need to delete the endpoint to stop charges from accumulating. See the clean up step at the end of this notebook.__

# In[ ]:


from sagemaker.jumpstart.model import JumpStartModel

embedding_model_id, embedding_model_version = "huggingface-textembedding-all-MiniLM-L6-v2", "*"
model = JumpStartModel(model_id=embedding_model_id, model_version=embedding_model_version)
embedding_predictor = model.deploy()

# In[16]:


# embedding_model_endpoint_name = "hf-textembedding-all-minilm-l6-v2-2023-09-07-22-28-08-201"

# In[ ]:


#this is the model endpoint NAME, not the ARN
embedding_model_endpoint_name = embedding_predictor.endpoint_name
embedding_model_endpoint_name

# To use your SageMaker model endpoints, you need to have a set of credentials. This section will assume them from your SageMaker Studio session.

# In[55]:


import boto3
aws_region = boto3.Session().region_name

# ## Creating and Populating the Vector Database

# Next you need to set up how to process the embeddings for the input documents.
# 
# The provided CustomEmbeddingsContentHandler class has a set of functions, transform_input and transform_output, for porcessing data going into and out of the embedding model.
# 
# With the content handler defined, you will then use the SageMakerEndpointEmbeddings class from LangChain to create an embeddings object that corresponds to your hosted embeddings model along with the appropriate content handler for processing its inputs/outputs.

# In[56]:


from typing import Dict, List
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
import json


class CustomEmbeddingsContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"text_inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["embedding"]


embeddings_content_handler = CustomEmbeddingsContentHandler()

embeddings = SagemakerEndpointEmbeddings(
    endpoint_name=embedding_model_endpoint_name,
    region_name=aws_region,
    content_handler=embeddings_content_handler,
)

# After downloading we can load the documents with the help of [DirectoryLoader from PyPDF available under LangChain](https://python.langchain.com/en/latest/reference/modules/document_loaders.html) and splitting them into smaller chunks.
# 
# Note: The retrieved document/text should be large enough to contain enough information to answer a question; but small enough to fit into the LLM prompt. Also the embeddings model has a limit of the length of input tokens limited to 512 tokens, which roughly translates to ~2000 characters. For the sake of this use-case we are creating chunks of roughly 1000 characters with an overlap of 100 characters using [RecursiveCharacterTextSplitter](https://python.langchain.com/en/latest/modules/indexes/text_splitters/examples/recursive_text_splitter.html).

# We had 3 PDF documents which have been split into smaller ~500 chunks.
# 
# Now we can see how a sample embedding would look like for one of those chunks

# In[57]:


query_embedding = np.array(embeddings.embed_query(docs[0].page_content))
np.array(query_embedding)

# The below function will establish a connection with OpenSearch Serverless, create a new index, create embeddings for the documents and then store the embeddings in OpenSearch serverless. For details on documentation refer this link: https://python.langchain.com/docs/integrations/vectorstores/opensearch
# 
# *Note: Wait for a minute or two after the below command to excute, before the new index can be queried.*

# In[58]:


index_name = "v2-try-again-sagemaker-embedding-384-opensearch-serverless-demo"
vector_size = 384

# In[59]:


# TODO - Direct langchain integration with version 0.0.245 gives timeout error, therefore, commenting the following code. 

# from langchain.vectorstores import OpenSearchVectorSearch

# docsearch = OpenSearchVectorSearch.from_documents(
#     docs,
#     bedrock_embeddings,
#     opensearch_url=host,
#     http_auth=awsauth,
#     timeout = 300,
#     use_ssl = True,
#     verify_certs = True,
#     connection_class = RequestsHttpConnection,
#     index_name="bedrock-aos-irs-index2",
#     engine="faiss",
#     bulk_size=len(docs)
# )

# In[60]:


# create a new index
index_body = {
    "settings": {
        "index.knn": True
  },
  'mappings': {
    'properties': {
      "title": { "type": "text", "fields": { "keyword": { "type": "keyword" } } }, #the field will be title.keyword and the data type will be keyword, this will act as sub field for
      "v_title": { "type": "knn_vector", "dimension": vector_size },
    }
  }
}

client.indices.create(
  index=index_name, 
  body=index_body
)

# In[61]:


# python code to view schema for OpenSearch Serverless. 
client.indices.get_mapping(index_name)

# In[62]:


actions =[]
bulk_size = 0
action = {"index": {"_index": index_name}}


# # Prepare bulk request
# actions.append(action)
# actions.append(json_data.copy())

# In[63]:


# TODO: Review this logic. Not sure if it works properly

# In[64]:


# Bulk API to ingest documents in OSS.
# it will take about 5 mins to ingest the 503 vectors
for document in docs: 
    sample_embedding = np.array(embeddings.embed_query(document.page_content))    
#    print(sample_embedding)
    actions.append(action)
    json_data = {
        "title" : document.page_content,
        "v_title" : sample_embedding
    }
    actions.append(json_data)
    bulk_size+=1
    if(bulk_size > 200 ):
        client.bulk(body=actions)
        print(f"bulk request sent with size: {bulk_size}")
        bulk_size = 0

# In[65]:


#print(actions)        
        
# ingest remaining documents
print("remaining documents: ", bulk_size)
client.bulk(body=actions)

# Following the similar pattern embeddings could be generated for the entire corpus and stored in a vector store.
# **⚠️⚠️⚠️ NOTE: it might take few minutes to run the following cell ⚠️⚠️⚠️**

# ### Question Answering
# 
# Now that we have our vector store in place, we can start asking questions.

# In[66]:


query = "How has Amazon evolved?"

# The first step would be to create an embedding of the query such that it could be compared with the documents

# In[67]:


query_embedding = np.array(embeddings.embed_query(query))
np.array(query_embedding)

# In[79]:


import time

time.sleep(120) # wait a couple minutes for OpenSearch to update the indexes

# In[80]:


query_os = {
  "size": 3,
  "fields": ["title"],
  "_source": False,
  "query": {
    "knn": {
      "v_title": {
        "vector": query_embedding,
        "k": vector_size
      }
    }
  }
}

relevant_documents = client.search(
    body = query_os,
    index = index_name
)

# In[81]:


relevant_documents

# We can use this embedding of the query to then fetch relevant documents.
# Now our query is represented as embeddings we can do a similarity search of our query against our data store providing us with the most relevant information.

# In[82]:


print(len(relevant_documents["hits"]["hits"]))
print("--------------------")
context = " "
for i, rel_doc in enumerate(relevant_documents["hits"]["hits"]):
    print(f'## Document {i+1}: {relevant_documents["hits"]["hits"][i]["fields"]["title"][0]}.......')
    print('---')
    context += relevant_documents["hits"]["hits"][i]["fields"]["title"][0]

# In[83]:


parameters = {
    "maxTokenCount":512,
    "stopSequences":[],
    "temperature":0,
    "topP":0.9
    }

# In[84]:


query

# In[85]:


context

# In[86]:


prompt = f"""Answer the below question based on the context provided. If the answer is not in the context, say "I don't know, answer not found in the documents".

{context}

{query}
"""

# In[ ]:


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_checkpoint = "NousResearch/Llama-2-7b-hf" # not gated
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForCausalLM.from_pretrained(model_checkpoint) # TODO use bfloat16 or int8 

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# In[ ]:


pipe(prompt)

# ## Conclusion
# Congratulations on completing this moduel on retrieval augmented generation! This is an important technique that combines the power of large language models with the precision of retrieval methods. By augmenting generation with relevant retrieved examples, the responses we recieved become more coherent, consistent and grounded. You should feel proud of learning this innovative approach. I'm sure the knowledge you've gained will be very useful for building creative and engaging language generation systems. Well done!
# 
# In the above implementation of RAG based Question Answering we have explored the following concepts and how to implement them using HuggingFace and SageMaker JumpStart models with LangChain integration.
# 
# - Loading documents and generating embeddings to create a vector store
# - Retrieving documents to the question
# - Preparing a prompt which goes as input to the LLM
# - Present an answer in a human friendly manner
# 
# ### Take-aways
# - Experiment with different Vector Stores
# - Explore options such as persistent storage of embeddings and document chunks
# - Integration with enterprise data stores
# 
# # Thank You
