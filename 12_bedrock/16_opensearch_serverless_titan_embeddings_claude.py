#!/usr/bin/env python
# coding: utf-8

# # Maximizing AI Potentials: Leveraging Foundational Models from Amazon Bedrock and Amazon OpenSearch Serverless as Vector Engine
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 
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
# To explain this architecture pattern we are using the Amazon shareholder letters for a few years.

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
# 
# 
# 
# Then begin with instantiating the LLM and the Embeddings model. Here we are using Amazon Titan to demonstrate the use case.
# 
# Note: It is possible to choose other models available with Bedrock. You can replace the `model_id` as follows to change the model.
# 
# `llm = Bedrock(model_id="...")`

# In[ ]:


%pip install langchain==0.0.305 --force-reinstall --quiet
%pip install pypdf==3.8.1 faiss-cpu==1.7.4 --force-reinstall --quiet

# In[ ]:


%pip install requests_aws4auth==1.2.3 opensearch-py==2.3.1 --force-reinstall --quiet

# In[ ]:


%pip install pydantic==1.10.13 --force-reinstall --quiet

# In[ ]:


%pip install sqlalchemy==2.0.21 --force-reinstall --quiet

# In[ ]:


#### Un comment the following lines to run from your local environment outside of the AWS account with Bedrock access

# import os
# os.environ['BEDROCK_ASSUME_ROLE'] = '<enter role>'
# os.environ['AWS_PROFILE'] = '<aws-profile>'

# In[ ]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# ### Create Amazon OpenSearch Serverless Index
# 
# To create it by using AWS console, following [this](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-manage.html) documentation.
# 
# Steps:
# 
# 1. On Amazon OpenSearch Serverless page, fill the fields following the template:
# 
# ![step-1](images/aoss-creation-1.png)
# 
# 2. Keep other fields as default: 
# 
# ![step-2](images/aoss-creation-2.png)
# 
# 3. Click on Next and on Summary page, click on Create.
# 
# 4. Check if it's created:

# In[ ]:


oss = boto3.client('opensearchserverless')

oss.list_collections()

# In[ ]:


# change here with the ID returned from previous command
collect_id = '<your-collection-id>'

# In[ ]:


import time

collection = oss.batch_get_collection(ids=[collect_id])

msg = "Status: {}. Collection: {} of type: {}"

while True:
    status = collection['collectionDetails'][0]['status']
    name = collection['collectionDetails'][0]['name']
    ctype = collection['collectionDetails'][0]['type']
    print(msg.format(status, name, ctype))
    if status != 'CREATING': break
    time.sleep(60)

# In[ ]:


host = collection['collectionDetails'][0]['collectionEndpoint']
host = host + ':443'
host

# ### Add OpenSearch Serverless Permissions
# 
# After index creation, it's necessary to add necessary permissions to add and to manage index and also to add data inside this collection.
# 
# To do so, on OpenSearch Serverless console, click into your collection, then on Data Access click on "Manage Data Access".
# 
# On permission management, make sure your role has following permissions and click on create (to add a new permission):
# 
# ![permissions](images/aoss-permission-1.png)

# You've setted Amazon OpenSearch Policy. Now it's important to setup your role to make sure it has necessary policies to call Amazon OpenSearch Serverless:
# 
# Sample IAM policy for SageMaker:
# 
# ```Json
# {
#     "Version": "2012-10-17",
#     "Statement": [
#         {
#             "Sid": "VisualEditor0",
#             "Effect": "Allow",
#             "Action": [
#                 "aoss:APIAccessAll",
#                 "aoss:ListCollections",
#                 "aoss:BatchGetCollection"
#             ],
#             "Resource": "*"
#         },
#         {
#             "Sid": "VisualEditor1",
#             "Effect": "Allow",
#             "Action": [
#                 "aoss:CreateIndex",
#                 "aoss:DeleteIndex",
#                 "aoss:UpdateIndex",
#                 "aoss:DescribeIndex",
#                 "aoss:ReadDocument",
#                 "aoss:WriteDocument"
#             ],
#             "Resource": "arn:aws:aoss:<region>:<accoount-id>:collection/*"
#         }
#     ]
# }
# ```

# ### Opensearch setup

# In[ ]:


from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import json

service = 'aoss'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,
session_token=credentials.token)

# ### Langchain setup
# 
# We created an instance of the Bedrock classes for the LLM and the embedding models. In this example we are showing an example with "titan" model from Amazon, and "claude" model from Anthropic.

# In[ ]:


# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# - create the Anthropic Model
claude_llm = Bedrock(model_id="anthropic.claude-v2", 
                     client=bedrock_runtime, 
                     model_kwargs={'max_tokens_to_sample':200})
titan_llm = Bedrock(model_id= "amazon.titan-text-express-v1", 
                    client=bedrock_runtime)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock_runtime)

# ### Data Preparation
# Let's first download some of the files to build our document store. For this example we will be using public Amazon Shareholder Letters.

# In[ ]:


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

# In[ ]:


from pypdf import PdfReader, PdfWriter
import glob

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

# In[ ]:


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

# In[ ]:


avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
print(f'Average length among {len(documents)} documents loaded is {avg_doc_length(documents)} characters.')
print(f'After the split we have {len(docs)} documents as opposed to the original {len(documents)}.')
print(f'Average length among {len(docs)} documents (after split) is {avg_doc_length(docs)} characters.')

# After downloading we can load the documents with the help of [DirectoryLoader from PyPDF available under LangChain](https://python.langchain.com/en/latest/reference/modules/document_loaders.html) and splitting them into smaller chunks.
# 
# Note: The retrieved document/text should be large enough to contain enough information to answer a question; but small enough to fit into the LLM prompt. Also the embeddings model has a limit of the length of input tokens limited to 512 tokens, which roughly translates to ~2000 characters. For the sake of this use-case we are creating chunks of roughly 1000 characters with an overlap of 100 characters using [RecursiveCharacterTextSplitter](https://python.langchain.com/en/latest/modules/indexes/text_splitters/examples/recursive_text_splitter.html).

# We had 3 PDF documents which have been split into smaller ~500 chunks.
# 
# Now we can see how a sample embedding would look like for one of those chunks

# In[ ]:


query_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
np.array(query_embedding)

# ### Index Data and query it
# 
# The below function will establish a connection with OpenSearch Serverless, create a new index, create embeddings for the documents and then store the embeddings in OpenSearch serverless. For details on documentation refer this link: https://python.langchain.com/docs/integrations/vectorstores/opensearch
# 
# *Note: Wait for a minute or two after the below command to excute, before the new index can be queried.*

# In[ ]:


from langchain.vectorstores import OpenSearchVectorSearch

docsearch = OpenSearchVectorSearch.from_documents(
    docs, 
    bedrock_embeddings, 
    opensearch_url=host,
    bulk_size=len(docs),
    http_auth=awsauth,
    timeout = 300,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    index_name="bedrock-opensearch-serverless-amazon-shareholder",
    engine="faiss",
)

# In[ ]:


## If Index has already been created, uncomment and run following cell:

#docsearch = OpenSearchVectorSearch(
#    opensearch_url=host,
#    index_name="bedrock-opensearch-serverless-amazon-shareholder",
#    embedding_function=bedrock_embeddings, 
#    http_auth=awsauth,
#    timeout = 300,
#    use_ssl = True,
#    verify_certs = True,
#    connection_class = RequestsHttpConnection,
#    is_aoss=True
#)

# In[ ]:


from langchain.indexes.vectorstore import VectorStoreIndexWrapper

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=docsearch)

# In[ ]:


query = "How has AWS evolved?"

# In[ ]:


%%time
answer = wrapper_store_faiss.query(question=query, llm=claude_llm)
print(answer)

# In[ ]:


%%time
answer = wrapper_store_faiss.query(question=query, llm=titan_llm)
print(answer)

# ## Conclusion
# Congratulations on completing this moduel on retrieval augmented generation! This is an important technique that combines the power of large language models with the precision of retrieval methods. By augmenting generation with relevant retrieved examples, the responses we recieved become more coherent, consistent and grounded. You should feel proud of learning this innovative approach. I'm sure the knowledge you've gained will be very useful for building creative and engaging language generation systems. Well done!
# 
# In the above implementation of RAG based Question Answering we have explored the following concepts and how to implement them using Amazon Bedrock and it's LangChain integration.
# 
# - Loading documents and generating embeddings to create a vector store
# - Retrieving documents to the question
# - Preparing a prompt which goes as input to the LLM
# - Present an answer in a human friendly manner
# 
# ### Take-aways
# - Experiment with different Vector Stores
# - Leverage various models available under Amazon Bedrock to see alternate outputs
# - Explore options such as persistent storage of embeddings and document chunks
# - Integration with enterprise data stores
# 
# # Thank You
