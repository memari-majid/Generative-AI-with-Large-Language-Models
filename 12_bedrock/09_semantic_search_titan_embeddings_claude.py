#!/usr/bin/env python
# coding: utf-8

# # Retrieval Augmented Question & Answering with Amazon Bedrock using LangChain
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 
# ### RAG Overview
# ![RAG Overview](images/rag-overview.png)
# 
# ### Challenges
# 
# When trying to solve a Question Answering task over a larger document corpus with the help of LLMs we need to master the following challenges (amongst others):
# - How to manage large document(s) that exceed the token limit
# - How to find the document(s) relevant to the question being asked
# 
# ### Infusing knowledge into LLM-powered systems
# 
# We have two primary [types of knowledge for LLMs](https://www.pinecone.io/learn/langchain-retrieval-augmentation/): 
# - **Parametric knowledge**: refers to everything the LLM learned during training and acts as a frozen snapshot of the world for the LLM. 
# - **Source knowledge**: covers any information fed into the LLM via the input prompt. 
# 
# When trying to infuse knowledge into a generative AI - powered application we need to choose which of these types to target. Fine-tuning, explored in other workshops, deals with elevating the parametric knowledge through fine-tuning. Since fine-tuning is a resouce intensive operation, this option is well suited for infusing static domain-specific information like domain-specific langauage/writing styles (medical domain, science domain, ...) or optimizing performance towards a very specific task (classification, sentiment analysis, RLHF, instruction-finetuning, ...). 
# 
# In contrast to that, targeting the source knowledge for domain-specific performance uplift is very well suited for all kinds of dynamic information, from knowledge bases in structured and unstructured form up to integration of information from live systems. This Lab is about retrieval-augmented generation, a common design pattern for ingesting domain-specific information through the source knowledge. It is particularily well suited for ingestion of information in form of unstructured text with semi-frequent update cycles. 
# 
# In this notebook we explain how to utilize the RAG (retrieval-agumented generation) pattern originating from [this](https://arxiv.org/pdf/2005.11401.pdf) paper published by Lewis et al in 2021. It is particularily useful for Question Answering by finding and leveraging the most useful excerpts of documents out of a larger document corpus providing answers to the user questions.
# 
# #### Prepare documents
# ![Embeddings](./images/embeddings_lang.png)
# 
# Before being able to answer the questions, the documents must be processed and a stored in a document store index
# - Load the documents
# - Process and split them into smaller chunks
# - Create a numerical vector representation of each chunk using Amazon Bedrock Titan Embeddings model
# - Create an index using the chunks and the corresponding embeddings
# #### Ask question
# ![Question](./images/chatbot_lang.png)
# 
# When the documents index is prepared, you are ready to ask the questions and relevant documents will be fetched based on the question being asked. Following steps will be executed.
# - Create an embedding of the input question
# - Compare the question embedding with the embeddings in the index
# - Fetch the (top N) relevant document chunks
# - Add those chunks as part of the context in the prompt
# - Send the prompt to the model under Amazon Bedrock
# - Get the contextual answer based on the documents retrieved

# ## Usecase
# #### Dataset
# In this example, you will use several years of Amazon's Letter to Shareholders as a text corpus to perform Q&A on.

# ## Implementation
# In order to follow the RAG approach this notebook is using the LangChain framework where it has integrations with different services and tools that allow efficient building of patterns such as RAG. We will be using the following tools:
# 
# - **LLM (Large Language Model)**: Anthropic Claude available through Amazon Bedrock
# 
#   This model will be used to understand the document chunks and provide an answer in human friendly manner.
# - **Embeddings Model**: Amazon Titan Embeddings available through Amazon Bedrock
# 
#   This model will be used to generate a numerical representation of the textual documents
# 
# - **Document Loader**: PDF Loader available through LangChain for PDFs
# 
#   These are loaders that can load the documents from a source, for the sake of this notebook we are loading the sample files from a local path. This could easily be replaced with a loader to load documents from enterprise internal systems.
# 
# - **Vector Store**: FAISS available through LangChain
#   In this notebook we are using this in-memory vector-store to store both the embeddings and the documents. In an enterprise context this could be replaced with a persistent store such as AWS OpenSearch, RDS Postgres with pgVector, ChromaDB, Pinecone or Weaviate.
# 
# - **Index**: VectorIndex
#   The index helps to compare the input embedding and the document embeddings to find relevant document.
# 
# - **Wrapper**: wraps index, vector store, embeddings model and the LLM to abstract away the logic from the user.
# 
# ### Setup
# To run this notebook you would need to install 2 more dependencies, [PyPDF](https://pypi.org/project/pypdf/) and [FAISS vector store](https://github.com/facebookresearch/faiss).
# 
# Then begin with instantiating the LLM and the Embeddings model. Here we are using Anthropic Claude to demonstrate the use case.
# 
# Note: It is possible to choose other models available with Bedrock. You can replace the `model_id` as follows to change the model.
# 
# `llm = Bedrock(model_id="...")`

# In[2]:


# %pip install --force-reinstall boto3 --quiet

# In[3]:


# %pip install langchain==0.0.309 --force-reinstall --quiet
# %pip install pypdf==3.8.1 faiss-cpu==1.7.4 --force-reinstall --quiet

# In[4]:


# %pip install tiktoken==0.4.0 --force-reinstall --quiet

# In[5]:


# %pip install sqlalchemy==2.0.21 --force-reinstall --quiet

# Uncomment the following lines to run from your local environment outside of the AWS account with Bedrock access. If you are carrying the lab out in Amazon SageMaker Studio, you are set without. 

# In[6]:


#import os
#os.environ['BEDROCK_ASSUME_ROLE'] = '<YOUR_VALUES>'
#os.environ['AWS_PROFILE'] = 'bedrock-user'

# In[7]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# In[8]:


from utils.TokenCounterHandler import TokenCounterHandler

token_counter = TokenCounterHandler()

# ### Setup langchain
# 
# We create an instance of the Bedrock classes for the LLM and the embedding models. At the time of writing, Bedrock supports one embedding model and therefore we do not need to specify any model id. To be able to compare token consumption across the different RAG-approaches shown in the workshop labs we use langchain callbacks to count token consumption.

# In[9]:


# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# - create the Anthropic Model
llm = Bedrock(model_id="anthropic.claude-v2", 
              client=bedrock_runtime, 
              model_kwargs={
                  'max_tokens_to_sample': 200
              }, 
              callbacks=[token_counter])

# - create the Titan Embeddings Model
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock_runtime)

# ### Data Preparation
# Let's first download some of the files to build our document store.
# 
# In this example, you will use several years of Amazon's Letter to Shareholders as a text corpus to perform Q&A on.

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

# We had 3 PDF documents and one txt file which have been split into smaller ~500 chunks.
# 
# Now we can see how a sample embedding would look like for one of those chunks.

# In[14]:


sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
print("Sample embedding of a document chunk: ", sample_embedding)
print("Size of the embedding: ", sample_embedding.shape)

# Following the very same approach embeddings can be generated for the entire corpus and stored in a vector store.
# 
# This can be easily done using [FAISS](https://github.com/facebookresearch/faiss) implementation inside [LangChain](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/faiss.html) which takes  input the embeddings model and the documents to create the entire vector store. Using the Index Wrapper we can abstract away most of the heavy lifting such as creating the prompt, getting embeddings of the query, sampling the relevant documents and calling the LLM. [VectorStoreIndexWrapper](https://python.langchain.com/en/latest/modules/indexes/getting_started.html#one-line-index-creation) helps us with that.
# 
# **⚠️⚠️⚠️ NOTE: it might take few minutes to run the following cell ⚠️⚠️⚠️**

# In[15]:


from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

vectorstore_faiss = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

# ### Question Answering
# 
# Now that we have our vector store in place, we can start asking questions.

# In[16]:


query = "How has AWS evolved?"

# The first step would be to create an embedding of the query such that it could be compared with the documents

# In[17]:


query_embedding = vectorstore_faiss.embedding_function(query)
np.array(query_embedding)

# We can use this embedding of the query to then fetch relevant documents.
# Now our query is represented as embeddings we can do a similarity search of our query against our data store providing us with the most relevant information.

# In[20]:


relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)
print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
print('----')
for i, rel_doc in enumerate(relevant_documents):
    print(f'## Document {i+1}: {rel_doc.page_content}.......')
    print('---')

# Now we have the relevant documents, it's time to use the LLM to generate an answer based on these documents. 
# 
# We will take our inital prompt, together with our relevant documents which were retreived based on the results of our similarity search. We then by combining these create a prompt that we feed back to the model to get our result. At this point our model should give us highly informed information on how we can change the tire of our specific car as it was outlined in our manual.
# 
# LangChain provides an abstraction of how this can be done easily.

# ### Quick way
# You have the possibility to use the wrapper provided by LangChain which wraps around the Vector Store and takes input the LLM.
# This wrapper performs the following steps behind the scences:
# - Takes input the question
# - Create question embedding
# - Fetch relevant documents
# - Stuff the documents and the question into a prompt
# - Invoke the model with the prompt and generate the answer in a human readable manner.

# In[21]:


answer = wrapper_store_faiss.query(question=query, llm=llm)
print(answer)

# Let's ask a different question:

# In[22]:


query_2 = "Why is Amazon successful?"

# In[23]:


answer_2 = wrapper_store_faiss.query(question=query_2, llm=llm)
print(answer_2)

# ### Customisable option
# In the above scenario you explored the quick and easy way to get a context-aware answer to your question. Now let's have a look at a more customizable option with the help of [RetrievalQA](https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa.html) where you can customize how the documents fetched should be added to prompt using `chain_type` parameter. Also, if you want to control how many relevant documents should be retrieved then change the `k` parameter in the cell below to see different outputs. In many scenarios you might want to know which were the source documents that the LLM used to generate the answer, you can get those documents in the output using `return_source_documents` which returns the documents that are added to the context of the LLM prompt. `RetrievalQA` also allows you to provide a custom [prompt template](https://python.langchain.com/en/latest/modules/prompts/prompt_templates/getting_started.html) which can be specific to the model.
# 
# Note: In this example we are using Anthropic Claude as the LLM under Amazon Bedrock, this particular model performs best if the inputs are provided under `Human:` and the model is requested to generate an output after `Assistant:`. In the cell below you see an example of how to control the prompt such that the LLM stays grounded and doesn't answer outside the context.

# In[24]:


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Assistant:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
    callbacks=[token_counter]
)

# In[25]:


query = "How did AWS evolve?"
result = qa({"query": query})
print(result['result'])

print(f"\n{result['source_documents']}")

# In[26]:


query = "Why is Amazon successful?"
result = qa({"query": query})
print(result['result'])

print(f"\n{result['source_documents']}")

# In[27]:


query = "What business challenges has Amazon experienced?"
result = qa({"query": query})
print(result['result'])

print(f"\n{result['source_documents']}")

# In[28]:


query = "How was Amazon impacted by COVID-19?"

result = qa({"query": query})

print(result['result'])

print(f"\n{result['source_documents']}")

# ## Conclusion
# Congratulations on completing this moduel on retrieval augmented generation! This is an important technique that combines the power of large language models with the precision of retrieval methods. By augmenting generation with relevant retrieved examples, the responses we recieved become more coherent, consistent and grounded. You should feel proud of learning this innovative approach. I'm sure the knowledge you've gained will be very useful for building creative and engaging language generation systems. Well done!
# 
# In the above implementation of RAG based Question Answering we have explored the following concepts and how to implement them using Amazon Bedrock and it's LangChain integration.
# 
# - Loading documents of different kind and generating embeddings to create a vector store
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

# In[ ]:



