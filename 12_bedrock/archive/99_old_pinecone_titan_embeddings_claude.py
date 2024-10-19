#!/usr/bin/env python
# coding: utf-8

# # Retrieval Augmented Question & Answering with Amazon Bedrock using LangChain & pinecone
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 

# ### Context
# Previously we saw that the model told us how to to change the tire, however we had to manually provide it with the relevant data and provide the contex ourselves. We explored the approach to leverage the model availabe under Bedrock and ask questions based on it's knowledge learned during training as well as providing manual context. While that approach works with short documents or single-ton applications, it fails to scale to enterprise level question answering where there could be large enterprise documents which cannot all be fit into the prompt sent to the model. 
# 
# ### Pattern
# We can improve upon this process by implementing an architecure called Retreival Augmented Generation (RAG). RAG retrieves data from outside the language model (non-parametric) and augments the prompts by adding the relevant retrieved data in context. 
# 
# In this notebook we explain how to approach the pattern of Question Answering to find and leverage the documents to provide answers to the user questions.
# 
# ### Challenges
# - How to manage large document(s) that exceed the token limit
# - How to find the document(s) relevant to the question being asked
# 
# ### Proposal
# To the above challenges, this notebook proposes the following strategy
# 
# #### Prepare documents
# <img src="./images/Embeddings_pinecone_lang.png" width="800"/>
# 
# Before being able to answer the questions, the documents must be processed and a stored in a document store index
# - Load the documents
# - Process and split them into smaller chunks
# - Create a numerical vector representation of each chunk using Amazon Bedrock Titan Embeddings model
# - Create an index using the chunks and the corresponding embeddings
# 
# #### Ask question
# <img src="./images/53-rag-with-pinecone.png" width="800"/>
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
# To explain this architecture pattern we are using the Amazon shareholder letters for a few years.

# ## Implementation
# In order to follow the RAG approach this notebook is using the LangChain framework where it has integrations with different services and tools that allow efficient building of patterns such as RAG. We will be using the following tools:
# 
# - **LLM (Large Language Model)**: Anthropic Claude available through Amazon Bedrock
# 
#   This model will be used to understand the document chunks and provide an answer in human friendly manner.
# - **Embeddings Model**: Amazon Titan Embeddings available through Amazon Bedrock
# 
#   This model will be used to generate a numerical representation of the textual documents
# - **Document Loader**: PDF Loader available through LangChain
# 
#   This is the loader that can load the documents from a source, for the sake of this notebook we are loading the sample files from a local path. This could easily be replaced with a loader to load documents from enterprise internal systems.
# 
# - **Vector Store**: Pinecone Vector Database Free Tier available through pinecone.io.
# 
#   In this notebook we are using Pinecone to store both the embeddings and the documents. In an enterprise context this could be replaced with another persistent store such as AWS OpenSearch, RDS Postgres with pgVector, ChromaDB, Pinecone or Weaviate.
# - **Index**: VectorIndex
# 
#   The index helps to compare the input embedding and the document embeddings to find relevant document
# - **Wrapper**: wraps index, vector store, embeddings model and the LLM to abstract away the logic from the user.
# 
# Built with the help of ideas in this [notebook](https://www.pinecone.io/learn/series/langchain/langchain-retrieval-augmentation/) and this [notebook](01_qa_w_rag_claude.ipynb)

# ## Setup
# 
# Before running the rest of this notebook, you'll need to run the cells below to (ensure necessary libraries are installed and) connect to Bedrock.
# 
# For more details on how the setup works and ⚠️ **whether you might need to make any changes**, refer to the [Bedrock boto3 setup notebook](../00_Intro/bedrock_boto3_setup.ipynb) notebook.
# 
# In this notebook, we'll also need some extra dependencies:
# 
# - [Pinecone](http://pinecone.io), to store vector embeddings
# - [PyPDF](https://pypi.org/project/pypdf/), for handling PDF files

# In[ ]:


!pip install -U \
    langchain==0.0.305 \
    pypdf==3.15.2 \
    pinecone-client==2.2.4 \
    apache-beam==2.50.0 \
    datasets==2.14.5 \
    tiktoken==0.4.0 --force-reinstall --quiet

# In[ ]:


%pip install pydantic==1.10.13 --force-reinstall --quiet

# In[ ]:


%pip install sqlalchemy==2.0.21 --force-reinstall --quiet

# In[ ]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# ## Configure langchain
# 
# We begin with instantiating the LLM and the Embeddings model. Here we are using Anthropic Claude for text generation and Amazon Titan for text embedding.
# 
# Note: It is possible to choose other models available with Bedrock. You can replace the `model_id` as follows to change the model.
# 
# `llm = Bedrock(model_id="...")`

# In[ ]:


# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# - create the Anthropic Model
llm = Bedrock(
    model_id="anthropic.claude-v2", 
    client=bedrock_runtime, 
    model_kwargs={"max_tokens_to_sample": 200}
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock_runtime)

# ## Data Preparation

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

# In[ ]:


avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
print(f'Average length among {len(documents)} documents loaded is {avg_doc_length(documents)} characters.')
print(f'After the split we have {len(docs)} documents as opposed to the original {len(documents)}.')
print(f'Average length among {len(docs)} documents (after split) is {avg_doc_length(docs)} characters.')

# In[ ]:


docs[0]

# In[ ]:


sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
print("Sample embedding of a document chunk: ", sample_embedding)
print("Size of the embedding: ", sample_embedding.shape)

# Following the similar pattern embeddings could be generated for the entire corpus and stored in a vector store.
# 
# This can be easily done using [Pinecone](https://python.langchain.com/docs/integrations/vectorstores/pinecone) implementation inside [LangChain](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/faiss.html) which takes  input the embeddings model and the documents to create the entire vector store. Using the Index Wrapper we can abstract away most of the heavy lifting such as creating the prompt, getting embeddings of the query, sampling the relevant documents and calling the LLM. [VectorStoreIndexWrapper](https://python.langchain.com/en/latest/modules/indexes/getting_started.html#one-line-index-creation) helps us with that.
# 

# In[ ]:


import pinecone
import time
import os

# add index name from pinecone.io
index_name = ''
# add Pinecone API key from app.pinecone.io
api_key = os.environ.get("PINECONE_API_KEY") or "YOUR_API_KEY"
# set Pinecone environment - find next to API key in console
env = os.environ.get("PINECONE_ENVIRONMENT") or "YOUR_ENV"

pinecone.init(api_key=api_key, environment=env)


if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

pinecone.create_index(name=index_name, dimension=sample_embedding.shape[0], metric="dotproduct")
# wait for index to finish initialization
while not pinecone.describe_index(index_name).status["ready"]:
    time.sleep(1)

# In[ ]:


index = pinecone.Index(index_name)
index.describe_index_stats()

# In[ ]:


docs[0]

# **⚠️⚠️⚠️ NOTE: it might take few minutes to run the following cell ⚠️⚠️⚠️**

# In[ ]:


%%time

from langchain.vectorstores import Pinecone

docsearch = Pinecone.from_documents(docs, bedrock_embeddings, index_name=index_name)

# In[ ]:


index.describe_index_stats()

# ## LangChain Vector Store and Querying
# 
# We construct our index independently of LangChain. That’s because it’s a straightforward process, and it is faster to do this with the Pinecone client directly. However, we’re about to jump back into LangChain, so we should reconnect to our index via the LangChain library.

# In[ ]:


from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(index, bedrock_embeddings.embed_query, text_field)

# #### We can use the similarity search method to make a query directly and return the chunks of text without any LLM generating the response.

# In[ ]:


query = "How has AWS evolved?"

vectorstore.similarity_search(query, k=3)  # our search query  # return 3 most relevant docs

# #### All of these are relevant results, telling us that the retrieval component of our systems is functioning. The next step is adding our LLM to generatively answer our question using the information provided in these retrieved contexts.

# ## Generative Question Answering
# 
# In generative question-answering (GQA), we pass our question to the Claude-2 but instruct it to base the answer on the information returned from our knowledge base. We can do this in LangChain easily using the RetrievalQA chain.

# In[ ]:


from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# #### Let’s try this with our earlier query:

# In[ ]:


qa.run(query)

# ### The response we get this time is generated by our gpt-3.5-turbo LLM based on the retrieved information from our vector database.
# 
# We’re still not entirely protected from convincing yet false hallucinations by the model, they can happen, and it’s unlikely that we can eliminate the problem completely. However, we can do more to improve our trust in the answers provided.
# 
# An effective way of doing this is by adding citations to the response, allowing a user to see where the information is coming from. We can do this using a slightly different version of the RetrievalQA chain called RetrievalQAWithSourcesChain.

# In[ ]:


from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

# In[ ]:


qa_with_sources(query)

# Now we have answered the question being asked but also included the source of this information being used by the LLM.
# 
# We’ve learned how to ground Large Language Models with source knowledge by using a vector database as our knowledge base. Using this, we can encourage accuracy in our LLM’s responses, keep source knowledge up to date, and improve trust in our system by providing citations with every answer.
# 
# We can use this embedding of the query to then fetch relevant documents.
# Now our query is represented as embeddings we can do a similarity search of our query against our data store providing us with the most relevant information.

# ### Customizable option
# In the above scenario you explored the quick and easy way to get a context-aware answer to your question. Now let's have a look at a more customizable option with the helpf of [RetrievalQA](https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa.html) where you can customize how the documents fetched should be added to prompt using `chain_type` parameter. Also, if you want to control how many relevant documents should be retrieved then change the `k` parameter in the cell below to see different outputs. In many scenarios you might want to know which were the source documents that the LLM used to generate the answer, you can get those documents in the output using `return_source_documents` which returns the documents that are added to the context of the LLM prompt. `RetrievalQA` also allows you to provide a custom [prompt template](https://python.langchain.com/en/latest/modules/prompts/prompt_templates/getting_started.html) which can be specific to the model.
# 
# Note: In this example we are using Anthropic Claude as the LLM under Amazon Bedrock, this particular model performs best if the inputs are provided under `Human:` and the model is requested to generate an output after `Assistant:`. In the cell below you see an example of how to control the prompt such that the LLM stays grounded and doesn't answer outside the context.

# In[ ]:


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)
result = qa({"query": query})
print(result["result"])

# In[ ]:


result["source_documents"]
