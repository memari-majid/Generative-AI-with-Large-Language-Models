#!/usr/bin/env python
# coding: utf-8

# # Retrieval Augmented Generation (RAG) with LangChain

# In this example notebook, you will see how to perform basic Retrieval Augmented Generation (RAG) using a collection of Amazon's Letters to Shareholders to run basic Q&A.
# 
# This notebook does not have any specific CPU/GPU requirements, and was built using the `Data Science 3.0 Python 3` kernel.

# ## Dependencies

# Install the dependencies for this example:
# - LangChain: Framework for Orchestrating the RAG workflow
# - FAISS: In-Memory Vector Database for storing document embeddings
# - PyPDF: Python library for processing PDF documents 

# In[2]:


# %pip install langchain==0.0.309 --quiet --root-user-action=ignore
# %pip install faiss-cpu==1.7.4 --quiet --root-user-action=ignore
# %pip install pypdf==3.15.1 --quiet --root-user-action=ignore

# ## Fetching and Processing the Sample Data

# Next, fetch the sample data for this example. This section will download the publicly available Amazon Letters to Shareholders, that are provided yearly as a "Year in Review" of Amazon's business.
# 
# This will download the pdfs locally and store them in a `data` directory local to this notebook.

# In[3]:


# !mkdir -p ./data

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

# As a part of Amazon's peculiar culture, the CEO always attaches the original 1997 Letter to Shareholders to the current letter. To reduce the amount of processing necessary, reduce bias towards that year, and improve output, you will use PyPDF to remove those pages from each file and re-save it over the original.

# In[4]:


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


# Now that you have clean PDFs to work with, they need to be broken down into manageable pieces so you can provide the most relevant sections to the LLM as part of your RAG workflow. Here, you will iterate over all the documents and break them down into 512 character chunks with an overlap of 100 characters.
# 
# The `chunk_size` dictates the size of the documents that will be embedded and stored in the vector database.
# 
# The `chunk_overlap` dictates the amount of text that is used from a previous chunk when building the next one. This allows you to maintain some of the context between chunks.
# 
# The `RecursiveCharacterTextSplitter` attempts to split up text recursively using delimeters of `["\n\n", "\n", " ", ""]` until achieving the desired chunk size. This attempts to keep paragraphs/sentences/words together to allow for better semantic analysis.

# In[5]:


import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

documents = []

for idx, file in enumerate(filenames):
    loader = PyPDFLoader(data_root + file)
    document = loader.load()
    for document_fragment in document:
        document_fragment.metadata = metadata[idx]
        
    documents += document

# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 100,
)

docs = text_splitter.split_documents(documents)

print(f'# of Document Pages {len(documents)}')
print(f'# of Document Chunks: {len(docs)}')

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

# In[6]:


from sagemaker.jumpstart.model import JumpStartModel

embedding_model_id, embedding_model_version = "huggingface-textembedding-all-MiniLM-L6-v2", "*"
model = JumpStartModel(model_id=embedding_model_id, model_version=embedding_model_version)
embedding_predictor = model.deploy()

# In[7]:


#this is the model endpoint NAME, not the ARN
embedding_model_endpoint_name = embedding_predictor.endpoint_name
embedding_model_endpoint_name

# To use your SageMaker model endpoints, you need to have a set of credentials. This section will assume them from your SageMaker Studio session.

# In[8]:


import boto3
aws_region = boto3.Session().region_name

# ## Creating and Populating the Vector Database

# Next you need to set up how to process the embeddings for the input documents.
# 
# The provided CustomEmbeddingsContentHandler class has a set of functions, transform_input and transform_output, for porcessing data going into and out of the embedding model.
# 
# With the content handler defined, you will then use the SageMakerEndpointEmbeddings class from LangChain to create an embeddings object that corresponds to your hosted embeddings model along with the appropriate content handler for processing its inputs/outputs.

# In[9]:


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

# With our embeddings references ready, the next step is to actually process those document chunks into vectors and store them somewhere. This example uses a FAISS in-memory vector database, but there are many other options available.

# In[10]:


from langchain.schema import Document
from langchain.vectorstores import FAISS

# In[11]:


db = FAISS.from_documents(docs, embeddings)

# ## Running Vector Queries

# Now that you have a populated vector database, you can run queries against it to return relevant document chunks.
# 
# Start with a simple query that corresponds to the source material.

# In[12]:


query = "How has AWS evolved?"

# The results that come back from the `similarity_search_with_score` API are sorted by score from lowest to highest. The score value is represented by the [L-squared (or L2)](https://en.wikipedia.org/wiki/Lp_space) distance of each result. Lower scores are better, repesenting a shorter distance between vectors.

# In[13]:


results_with_scores = db.similarity_search_with_score(query)
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}\nScore: {score}\n\n")

# In[14]:


filter={"year": 2022}

results_with_scores = db.similarity_search_with_score(query,
  filter=filter)

for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}\nScore: {score}\n\n")


# ## Creating Prompts

# You've gotten results from your vector database, but currently they are just chunks of the original documents and some of them might not even contain the information you want to provide as an answer to your original query.
# 
# To generate the appropriate response, you will leverage a prompt template that takes the original question asked along with relevant context chunks from your vector database to generate a new response from your language generator model.
# 
# LangChain provides functionality to allow for easier creation and population of prompt templates. The template below has specific markup for LLaMa-2-chat, but also has placeholder values for `{context}` and `{question}`, which you will provide to fill out the template.

# In[15]:


from langchain.prompts import PromptTemplate

prompt_template = """
<s>[INST] <<SYS>>
Use the context provided to answer the question at the end. If you dont know the answer just say that you don't know, don't try to make up an answer.
<</SYS>>

Context:
----------------
{context}
----------------

Question: {question} [/INST]
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ## Preparing the LLM

# The next step is a process similar to the one you did earlier for the embedding model, but now for your LLM.
# 
# In the QAContentHandler class, you will see `transform_input` and `transform_output` functions to manipulate the inputs and outputs of your LLM.

# In[16]:


from typing import Dict

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
import json


class QAContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps(
            {"inputs" : [
                [
                    {
                        "role" : "system",
                        "content" : ""
                    },
                    {
                        "role" : "user",
                        "content" : prompt
                    }
                ]],
                "parameters" : {**model_kwargs}
            })
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]

qa_content_handler = QAContentHandler()

# Now you will deploy a SageMaker endpoint for language generation LLM. Afterward you will create an object pointed to that endpoint and provide inference parameters to the endpoint and model.
# 
# If you already have a LLM endpoint deployed, you can skip the following cell, and modify the `llm_model_endpoint_name` value to match your endpoint.

# __Note: running the following cell will deploy a SageMaker endpoint which takes a few minutes. You will need to delete the endpoint to stop charges from accumulating. See the clean up step at the end of this notebook.__

# In[ ]:


llm_model_id, llm_model_version = "meta-textgeneration-llama-2-7b-f", "2.*"
llm_model = JumpStartModel(model_id=llm_model_id, model_version=llm_model_version)
llm_predictor = llm_model.deploy()

# In[18]:


#this is the model endpoint NAME, not the ARN
llm_model_endpoint_name = llm_predictor.endpoint_name
llm_model_endpoint_name

# In[19]:


llm = SagemakerEndpoint(
        endpoint_name=llm_model_endpoint_name,
        region_name=aws_region,
        model_kwargs={"max_new_tokens": 1000, "top_p": 0.9, "temperature": 1e-11},
        endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
        content_handler=qa_content_handler
    )

# You can invoke this LLM object directly to get a baseline response without any contextual information provided. You'll notice the answer to the question `How has AWS evolved?` is more about __what__ AWS has done rather than a more internal take on how AWS has evolved. This is likely due to the corpus of data that the LLM was trained on which contained a large amount of articles from the internet.
# 
# Note that this is not a bad response by any stretch, but it might not be the response you're looking for.
# 
# You'll see how context can evolve the reponse in a moment.

# In[20]:


query = "How has AWS evolved?"
llm.predict(query)

# With the LLM endpoint object created, you are ready to create your first chain!
# 
# This chain is a simple example using LangChain's RetrievalQA chain, which will:
# - take a query as input
# - generate query embeddings
# - query the vector database for relevant document chunks based on the query embedding
# - inject the context and original query into the prompt template
# - invoke the LLM with the completed prompt
# - return the LLM result
# 
# The [`stuff` chain type](https://python.langchain.com/docs/modules/chains/document/stuff) simply takes the context documents and inserts them into the prompt.
# 
# By setting `return_source_documents` to `True`, the LLM responses will also contain the document chunks from the vector database, to illustrate where the context came from.

# In[21]:


qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type='stuff',
    retriever=db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Now that your chain is set up, you can supply queries to it and generate responses based on your source documents.
# 
# A few examples have been provided.

# In[22]:


query = "How has AWS evolved?"
result = qa_chain({"query": query})
print(f'Query: {result["query"]}\n')
print(f'Result: {result["result"]}\n')
print(f'Context Documents: ')
for srcdoc in result["source_documents"]:
      print(f'{srcdoc}\n')

# In[23]:


qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type='stuff',
    retriever=db.as_retriever(
        search_type="mmr", # Maximum Marginal Relevance (MMR)
        search_kwargs={"k": 3, "lambda_mult": 0.1}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Now that your chain is set up, you can supply queries to it and generate responses based on your source documents.
# 
# A few examples have been provided.

# In[24]:


query = "How has AWS evolved?"
result = qa_chain({"query": query})
print(f'Query: {result["query"]}\n')
print(f'Result: {result["result"]}\n')
print(f'Context Documents: ')
for srcdoc in result["source_documents"]:
      print(f'{srcdoc}\n')

# In[25]:


query = "Why is Amazon successful?"
result = qa_chain({"query": query})
print(f'Query: {result["query"]}\n')
print(f'Result: {result["result"]}\n')
print(f'Context Documents: ')
for srcdoc in result["source_documents"]:
      print(f'{srcdoc}\n')

# In[26]:


query = "What business challenges has Amazon experienced?"
result = qa_chain({"query": query})
print(f'Query: {result["query"]}\n')
print(f'Result: {result["result"]}\n')
print(f'Context Documents: ')
for srcdoc in result["source_documents"]:
      print(f'{srcdoc}\n')

# # Clean up

# Uncomment the `delete_endpoint` calls to remove the resources you created.

# In[ ]:


# sagemaker_client = boto3.client('sagemaker', region_name=aws_region)

# #Delete embedding endpoint
# sagemaker_client.delete_endpoint(EndpointName=embedding_model_endpoint_name)

# #Delete llm endpoint
# sagemaker_client.delete_endpoint(EndpointName=llm_model_endpoint_name)

# In[ ]:



