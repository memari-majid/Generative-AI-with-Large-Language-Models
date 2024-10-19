#!/usr/bin/env python
# coding: utf-8

# # Llama Document Summary Index With Amazon Bedrock Titan Model
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 
# 
# This demo showcases the document summary index, over IRS Forms.
# 
# The document summary index will extract a summary from each document and store that summary, as well as all nodes corresponding to the document.
# 
# Retrieval can be performed through the LLM or embeddings. We first select the relevant documents to the query based on their summaries. All retrieved nodes corresponding to the selected documents are retrieved.

# This notebook demonstrates invoking Bedrock models directly using the AWS SDK, but for later notebooks in the workshop you'll also need to install [LangChain](https://github.com/hwchase17/langchain):

# In[2]:


%pip install langchain==0.0.309 --force-reinstall --quiet
%pip install pypdf==3.15.2 --force-reinstall --quiet
%pip install llama-index==0.8.37 --force-reinstall --quiet
%pip install sentence_transformers==2.2.2 --force-reinstall --quiet

# In[3]:


%pip install pydantic==1.10.13 --force-reinstall --quiet

# In[4]:


%pip install sqlalchemy==2.0.21 --force-reinstall --quiet

# In[5]:


# needed for llamaindex
%pip install openai==0.28 --force-reinstall --quiet 

# In[6]:


import nest_asyncio

nest_asyncio.apply()

# In[7]:


from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    get_response_synthesizer,
    set_global_service_context
)
from llama_index.indices.document_summary import DocumentSummaryIndex

# ### Load Datasets

# In[8]:


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

# In[9]:


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


# Now that you have clean PDFs to work with, you will enrich your documents with metadata, then use a process called "chunking" to break up a larger document into small pieces. These small pieces will allow you to generate embeddings without surpassing the input limit of the embedding model.
# 
# In this example you will break the document into 1000 character chunks, with a 100 character overlap. This will allow your embeddings to maintain some of its context.

# In[10]:


docs = []
for filename in filenames:
    doc = SimpleDirectoryReader(input_files=[f"data/{filename}"]).load_data()
    doc[0].doc_id = filename.replace(".pdf", "")
    docs.extend(doc)

# ### Build Document Summary Index
# 
# We show two ways of building the index:
# - default mode of building the document summary index
# - customizing the summary query
# 

# In[11]:


#### Un comment the following lines to run from your local environment outside of the AWS account with Bedrock access

#import os
#os.environ['BEDROCK_ASSUME_ROLE'] = '<YOUR_VALUES>'
#os.environ['AWS_PROFILE'] = 'bedrock-user'

# In[12]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# In[13]:


from llama_index import LangchainEmbedding
from langchain.llms.bedrock import Bedrock # required until llama_index offers direct Bedrock integration
from langchain.embeddings.bedrock import BedrockEmbeddings

model_kwargs_titan = { 
    "maxTokenCount": 512,
    "stopSequences": [],
    "temperature":0.0,  
    "topP":0.5
}

llm = Bedrock(model_id="amazon.titan-text-express-v1",
              client=bedrock_runtime,
              model_kwargs=model_kwargs_titan)

embed_model = LangchainEmbedding(
  BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
)

service_context = ServiceContext.from_defaults(llm=llm, 
                                               embed_model=embed_model, 
                                               chunk_size=512)
set_global_service_context(service_context)

# ### This will take a few minutes. Please be patient.

# In[14]:


response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)
doc_summary_index = DocumentSummaryIndex.from_documents(
    docs,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
)

# In[15]:


print(len(docs))

# In[16]:


doc_summary_index.get_document_summary("AMZN-2022-Shareholder-Letter")

# In[17]:


doc_summary_index.storage_context.persist("llama_titan_index")

# In[18]:


from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="llama_titan_index")
doc_summary_index = load_index_from_storage(storage_context)

# ### Perform Retrieval from Document Summary Index
# 
# We show how to execute queries at a high-level. We also show how to perform retrieval at a lower-level so that you can view the parameters that are in place. We show both LLM-based retrieval and embedding-based retrieval using the document summaries.

# #### High-level Querying
# 
# Note: this uses the default, LLM-based form of retrieval

# In[19]:


from utils.llama_custom_parse_choice_select_answer_fn import custom_parse_choice_select_answer_fn

query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True, similarity_top_k=10,  verbose=False,
    parse_choice_select_answer_fn=custom_parse_choice_select_answer_fn
)

# In[20]:


response = query_engine.query("How has AWS evolved?")

print(response)

# #### LLM-based Retrieval

# In[21]:


from llama_index.indices.document_summary import DocumentSummaryIndexRetriever

# In[22]:


from utils.llama_custom_parse_choice_select_answer_fn import custom_parse_choice_select_answer_fn
retriever = DocumentSummaryIndexRetriever(
    doc_summary_index,
    # choice_select_prompt=choice_select_prompt,
    # choice_batch_size=choice_batch_size,
    # format_node_batch_fn=format_node_batch_fn,
    parse_choice_select_answer_fn=custom_parse_choice_select_answer_fn,
    # service_context=service_context
)

# In[23]:


retrieved_nodes = retriever.retrieve("According to the documents, How has AWS evolved?")
print(len(retrieved_nodes))

# In[24]:


for i, node in enumerate(retrieved_nodes):
    print(node.score)
    print(node.node.get_text())
    print("-----------------------------------------------------------------------------------------------------------------------------------")

# In[25]:


%%time
# use retriever as part of a query engine
from llama_index.query_engine import RetrieverQueryEngine

# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="refine", use_async=True)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
response = query_engine.query("How has AWS evolved?")
print(response)

# #### Embedding-based Retrieval

# In[26]:


from llama_index.indices.document_summary import DocumentSummaryIndexEmbeddingRetriever

# In[27]:


retriever = DocumentSummaryIndexEmbeddingRetriever(
    doc_summary_index,
    # choice_select_prompt=choice_select_prompt,
    # choice_batch_size=choice_batch_size,
    # format_node_batch_fn=format_node_batch_fn,
    # parse_choice_select_answer_fn=parse_choice_select_answer_fn,
    #service_context=service_context
)

# In[28]:


retrieved_nodes = retriever.retrieve("How has AWS evolved?")

# In[29]:


len(retrieved_nodes)

# In[30]:


for i, node in enumerate(retrieved_nodes):
    
    print(node.node.get_text())
    print("-----------------------------------------------------------------------------------------------------------------------------------")
    

# In[ ]:



