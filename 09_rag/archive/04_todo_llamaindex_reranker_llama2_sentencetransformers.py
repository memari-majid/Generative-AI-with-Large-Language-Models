#!/usr/bin/env python
# coding: utf-8

# # Using LLM-powered retreival and reranking - (Claude LLM + Bedrock Titan embedding)
# 
# ### Context
# 
# Utilizing LLM-driven retrieval has the potential to yield more pertinent documents compared to retrieval based on embeddings. However, this advantage comes at the expense of increased latency and expenses. We will demonstrate that employing embedding-based retrieval initially, followed by a secondary retrieval stage for reevaluation, can offer a balanced solution.
# 
# A recent surge in applications involving "Develop a chatbot using your data" has emerged in the past several months. This trend has been facilitated by frameworks such as LlamaIndex and LangChain. Many of these applications rely on a standard approach known as retrieval augmented generation (RAG):
# 
# 1) A vector store is employed to store unstructured documents (knowledge corpus).
# 2) When presented with a query, a retrieval model is utilized to fetch relevant documents from the corpus, followed by a synthesis model that generates a response.
# 3) The retrieval model retrieves the top-k documents based on the similarity of their embeddings to the query. It's important to note that the concept of top-k embedding-based semantic search has existed for over a decade and doesn't involve the use of LLM.
# 
# The utilization of embedding-based retrieval offers numerous advantages:
# 
# * Dot product calculations are swift and don't necessitate model invocations during query processing.
# * Although not flawless, embeddings can effectively capture the semantics of documents and queries. There's a subset of queries for which embedding-based retrieval yields highly relevant outcomes.
# 
# However, embedding-based retrieval can exhibit imprecision and return irrelevant context for the query due to various factors. This subsequently diminishes the quality of the overall RAG system, irrespective of the LLM's quality.
# 
# Addressing this challenge is not novel; existing information retrieval and recommendation systems have adopted a two-stage approach. The initial stage employs embedding-based retrieval with a higher top-k value to maximize recall while accepting a lower precision. Subsequently, the second stage utilizes a somewhat more computationally intensive process characterized by higher precision and lower recall (such as BM25) to "rerank" the initially retrieved candidates.
# 
# Delving into the shortcomings of embedding-based retrieval would require an entire series of blog posts. This current post serves as an initial exploration of an alternative retrieval technique and its potential to enhance embedding-based retrieval methodologies.
#  
# ![LLM retrival works](./images/arch.png)
# 
# ### LLM Retrieval and Reranking
# 
# LLM Retrieval and reranking strategy employs the LLM to determine the document(s) or sections of text that align with the provided query. The input prompt comprises a collection of potential documents, and the LLM is entrusted with choosing the pertinent group of documents while also assigning a score to gauge their relevance using an internal measurement.
# 
# 
# In this notebook we explain how to approach the retriever pattern of LLM-powered retrieval and reranking using Amazon Bedrock LLM and LlamaIndex
# 
# #### LlamaIndex
# LlamaIndex is a data framework for your LLM application. It provides the following tools:
# 
# * Offers data connectors to ingest your existing data sources and data formats (APIs, PDFs, docs, SQL, etc.)
# * Provides ways to structure your data (indices, graphs) so that this data can be easily used with LLMs.
# * Provides an advanced retrieval/query interface over your data: Feed in any LLM input prompt, get back retrieved context and knowledge-augmented output.
# * Allows easy integrations with your outer application framework (e.g. with LangChain, Flask, Docker, anything else).
# * LlamaIndex provides tools for both beginner users and advanced users. Our high-level API allows beginner users to use LlamaIndex to ingest and query their data in 5 lines of code. Our lower-level APIs allow advanced users to customize and extend any module (data connectors, indices, retrievers, query engines, reranking modules), to fit their needs.
# 
# ### LLM Used:
# We will be leveraging Bedrock - Anthropic Claude LLM and Bedrock Embedding (Titan model) for demonstration.
# 
# 

# ### Setup
# 
# We will first install the necessary libraries

# In[ ]:


!cd .. && ./download-dependencies.sh

# In[ ]:


import glob
import subprocess

botocore_whl_filename = glob.glob("../dependencies/botocore-*-py3-none-any.whl")[0]
boto3_whl_filename = glob.glob("../dependencies/boto3-*-py3-none-any.whl")[0]

subprocess.Popen(['pip', 'install', botocore_whl_filename, boto3_whl_filename, '--force-reinstall'], bufsize=1, universal_newlines=True)

# In[ ]:


%pip install pydantic==1.10.12 --force-reinstall 

# In[ ]:


# langchain==0.0.266 is required by llama-index==0.8.8
%pip install langchain==0.0.266 \
    pypdf==3.15.2 \
    llama-index==0.8.8 \
    sentence_transformers==2.2.2 --force-reinstall

# In[ ]:


import nest_asyncio

nest_asyncio.apply()

# In[ ]:


import sys

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    LLMPredictor,
    get_response_synthesizer,
    set_global_service_context,
    StorageContext,
    ListIndex
)
from llama_index.indices.postprocessor import LLMRerank
from llama_index.llms import OpenAI
from IPython.display import Markdown, display

# ### Setup langchain and llama index
# 
# In this step we will be creating of instance for LLM and embedding models. We will be using Claude and Titan models

# In[ ]:


from llama_index import LangchainEmbedding
from langchain.llms.bedrock import Bedrock 
from langchain.embeddings.bedrock import BedrockEmbeddings

model_kwargs_claude = {
    "temperature":0,
    "top_k":10, 
    "max_tokens_to_sample":512
}

llm = Bedrock(model_id="anthropic.claude-v2",
              model_kwargs=model_kwargs_claude)

embed_model = LangchainEmbedding(
    BedrockEmbeddings(model_id="amazon.titan-e1t-medium")
)

service_context = ServiceContext.from_defaults(llm=llm, 
                                               embed_model=embed_model, 
                                               chunk_size=512)
set_global_service_context(service_context)

# ### Load Datasets

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

# In[ ]:


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

# In[ ]:


#### Un comment the following lines to run from your local environment outside of the AWS account with Bedrock access

#import os
#os.environ['BEDROCK_ASSUME_ROLE'] = '<YOUR_VALUES>'
#os.environ['AWS_PROFILE'] = 'bedrock-user'

# In[ ]:


import os
import boto3
import json
import sys

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
boto3_bedrock = bedrock.get_bedrock_client(os.environ.get('BEDROCK_ASSUME_ROLE', None))

# In[ ]:


boto3_bedrock.list_foundation_models()

# In[ ]:


index = VectorStoreIndex.from_documents(docs,
    service_context=service_context)

# In[ ]:


nodes = service_context.node_parser.get_nodes_from_documents(docs)

# In[ ]:


# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# ## Retrieval

# In[ ]:


from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType

CLAUDE_CHOICE_SELECT_PROMPT_TMPL = (
    """Human: A list of documents is shown below. Each document has a number next to it along  with a summary of the document. A question is also provided. 
    Respond with the respective document number. You should consult to answer the question, in order of relevance, as well as the relevance score. 
    The relevance score is a number from 1-10 based on how relevant you think the document is to the question.\n"
    Do not include any documents that are not relevant to the question. 
    
    Example format: 
    Document 1:\n<summary of document 1>
    Document 2:\n<summary of document 2>
    ...\n\n
    Document 10:\n<summary of document 10>
    
    Question: <question>
    Answer:
    Doc: 9, Relevance: 7
    Doc: 3, Relevance: 4
    Doc: 7, Relevance: 3

    Let's try this now: 
    {context_str}

    Question: {query_str}
    Assistant: Answer:
    """
)
claude_choice_select_prompt = Prompt(
    CLAUDE_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)

# ## Retrieval

# In[ ]:


from llama_index.retrievers import VectorIndexRetriever
from llama_index.indices.query.schema import QueryBundle
import pandas as pd
from IPython.display import display, HTML

pd.set_option("display.max_colwidth", None)

def get_retrieved_nodes(
    query_str, vector_top_k=10, reranker_top_n=3, with_reranker=False
):
    query_bundle = QueryBundle(query_str)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
        choice_select_prompt=claude_choice_select_prompt

    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        # configure reranker
        reranker = LLMRerank(
            choice_batch_size=5, 
            top_n=reranker_top_n, 
            service_context=service_context, 
            choice_select_prompt=claude_choice_select_prompt

        )
        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

    return retrieved_nodes


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def visualize_retrieved_nodes(nodes) -> None:
    result_dicts = []
    for node in nodes:
        result_dict = {"Score": node.score, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    pretty_print(pd.DataFrame(result_dicts))

# Now, we will showcase how to do a two-stage pass for retrieval. Use embedding-based retrieval with a high top-k value in order to maximize recall and get a large set of candidate items. Then, use LLM-based retrieval to dynamically select the nodes that are actually relevant to the query.

# In[ ]:


retrieved_nodes1 = get_retrieved_nodes(
    "Human: How has AWS evolved?", vector_top_k=3, with_reranker=False
)

# In[ ]:


len(retrieved_nodes1)

# In[ ]:


for i, node in enumerate(retrieved_nodes1):
    print(node.score)
    print(node.node.get_text())
    print("-----------------------------------------------------------------------------------------------------------------------------------")
    

# In[ ]:


retrieved_nodes1_withreranker = get_retrieved_nodes(
    "Human: How has AWS evolved?",
    vector_top_k=3,
    reranker_top_n=1,
    with_reranker=True,
)

# In[ ]:


len(retrieved_nodes1_withreranker)

# In[ ]:


for i, node in enumerate(retrieved_nodes1_withreranker):
    print(node.score)
    print(node.node.get_text())
    print("-----------------------------------------------------------------------------------------------------------------------------------")
    

# In[ ]:


retrieved_nodes2 = get_retrieved_nodes(
    "Human: Why is Amazon successful?", vector_top_k=3, with_reranker=False
)

# In[ ]:


len(retrieved_nodes2)

# In[ ]:


for i, node in enumerate(retrieved_nodes2):
    print(node.score)
    print(node.node.get_text())
    print("-----------------------------------------------------------------------------------------------------------------------------------")
    

# In[ ]:


retrieved_nodes2_withreranker = get_retrieved_nodes(
    "Human: Why is Amazon successful?",
    vector_top_k=3,
    reranker_top_n=1,
    with_reranker=True,
)

# In[ ]:


len(retrieved_nodes2_withreranker)

# In[ ]:


for i, node in enumerate(retrieved_nodes2_withreranker):
    print(node.score)
    print(node.node.get_text())
    print("-----------------------------------------------------------------------------------------------------------------------------------")
    

# In[ ]:



