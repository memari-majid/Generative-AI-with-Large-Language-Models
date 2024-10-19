#!/usr/bin/env python
# coding: utf-8

# # Semantic Search with Metadata Filtering using FAISS and Amazon Bedrock
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 

# ## Install dependencies

# This notebook demonstrates invoking Bedrock models directly using the AWS SDK, but for later notebooks in the workshop you'll also need to install [LangChain](https://github.com/hwchase17/langchain):

# In[2]:


%pip install langchain==0.0.309 --force-reinstall

# In[3]:


%pip install pydantic==1.10.13 --force-reinstall --quiet

# In[4]:


%pip install sqlalchemy==2.0.21 --force-reinstall --quiet

# In this example, you will use [Facebook AI Similarity Search (Faiss)](https://faiss.ai/) as the vector database to store your embeddings. There are CPU or GPU options available, depending on your platform.

# In[5]:


%pip install faiss-cpu==1.7.4 # For CPU Installation
#%pip install faiss-gpu # For CUDA 7.5+ Supported GPU's.

# In this example, you will use several years of Amazon's Letter to Shareholders as a text corpus to perform Q&A on.
# 
# First you will download these files from the internet.

# In[6]:


%pip install pypdf==3.14.0

# In[7]:


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

# In[8]:


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

# In[9]:


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

# ## Create the boto3 client
# 
# Interaction with the Bedrock API is done via boto3 SDK. To create a the Bedrock client, we are providing an utility method that supports different options for passing credentials to boto3. 
# If you are running these notebooks from your own computer, make sure you have [installed the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) before proceeding.
# 
# 
# #### Use default credential chain
# 
# If you are running this notebook from a Sagemaker Studio notebook and your Sagemaker Studio role has permissions to access Bedrock you can just run the cells below as-is. This is also the case if you are running these notebooks from a computer whose default credentials have access to Bedrock
# 
# #### Use a different role
# 
# In case you or your company has setup a specific role to access Bedrock, you can specify such role by uncommenting the line `#os.environ['BEDROCK_ASSUME_ROLE'] = '<YOUR_VALUES>'` in the cell below before executing it. Ensure that your current user or role have permissions to assume such role.
# 
# #### Use a specific profile
# 
# In case you are running this notebooks from your own computer and you have setup the AWS CLI with multiple profiles and the profile which has access to Bedrock is not the default one, you can uncomment the line `#os.environ['AWS_PROFILE'] = '<YOUR_VALUES>'` and specify the profile to use.
# 
# #### Note about `langchain`
# 
# The Bedrock classes provided by `langchain` create a default Bedrock boto3 client. We recommend to explicitly create the Bedrock client using the instructions below, and pass it to the class instantiation methods using `client=bedrock_runtime`

# In[10]:


#### Un comment the following lines to run from your local environment outside of the AWS account with Bedrock access

#import os
#os.environ['BEDROCK_ASSUME_ROLE'] = '<YOUR_VALUES>'
#os.environ['AWS_PROFILE'] = 'bedrock-user'

# In[11]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# ## Building a FAISS vector database

# In this example, you will be using the Amazon Titan Embeddings Model from Amazon Bedrock to generate the embeddings for our FAISS vector database.

# The `TokenCounterHandler` callback function is a function you can utilize in your LLM objects and chains to generate reports on token count. It is supplied here as a utility class that will output the token counts at the end of your result chain, or can be attached to a LLM object and invoked manually.

# In[12]:


%pip install tiktoken==0.4.0 --force-reinstall

# In[13]:


from utils.TokenCounterHandler import TokenCounterHandler

token_counter = TokenCounterHandler()

# In[14]:


from langchain.embeddings import BedrockEmbeddings

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                               client=bedrock_runtime)

# Next you will import the Document and FAISS modules from Langchain. Using these modules will allow you to quickly generate embeddings through Amazon Bedrock and store them locally in your FAISS vector store.

# In[15]:


from langchain.schema import Document
from langchain.vectorstores import FAISS

# In this step you will process documents and prepare them to be converted to vectors for the vector store.

# Here you will use the from_documents function in the Langchain FAISS provider to build a vector database from your document embeddings.

# In[16]:


db = FAISS.from_documents(docs, embeddings)

# To avoid having to completely regenerate your embeddings all the time, you can save and load the vector store from the local filesystem. In the next section you will save the vector store locally, and reload it.

# In[17]:


db.save_local("faiss_claude_index")

# In[18]:


new_db = FAISS.load_local("faiss_claude_index", embeddings)

# In[19]:


db = new_db

# ## Similarity Searching

# Here you will set your search query, and look for documents that match.

# In[20]:


query = "How has AWS evolved?"

# ### Basic Similarity Search

# The results that come back from the `similarity_search_with_score` API are sorted by score from lowest to highest. The score value is represented by the [L-squared (or L2)](https://en.wikipedia.org/wiki/Lp_space) distance of each result. Lower scores are better, repesenting a shorter distance between vectors.

# In[21]:


results_with_scores = db.similarity_search_with_score(query)
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}\nScore: {score}\n\n")

# ### Similarity Search with Metadata Filtering
# Additionally, you can provide metadata to your query to filter the scope of your results. The `filter` parameter for search queries is a dictionary of metadata key/value pairs that will be matched to results to include/exclude them from your query.

# In[22]:


filter = dict(year=2022)

# In the next section, you will notice that your query has returned less results than the basic search, because of your filter criteria on the resultset.

# In[23]:


results_with_scores = db.similarity_search_with_score(query, filter=filter)
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}, Score: {score}\n\n")

# ### Top-K Matching
# 
# Top-K Matching is a filtering technique that involves a 2 stage approach.
# 
# 1. Perform a similarity search, returning the top K matches.
# 2. Apply your metadata filter on the smaller resultset.
# 
# Note: A caveat for Top-K matching is that if the value for K is too small, there is a chance that after filtering there will be no results to return.
# 
# Using Top-K matching requires 2 values:
# - `k`, the max number of results to return at the end of our query
# - `fetch_k`, the max number of results to return from the similarity search before applying filters
# 

# In[24]:


results = db.similarity_search(query, filter=filter, k=2, fetch_k=4)
for doc in results:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}\n\n")

# ### Maximal Marginal Relevance
# 
# Another measurement of results is Maximal Marginal Relevance (MMR). The focus of MMR is to minimize the redundancy of your search results while still maintaining relevance by re-ranking the results to provide both similarity and diversity.
# 
# In the next section you will use the `max_marginal_relevance_search` API to run the same query as in the Metadata Filtering section, but with reranked results.

# In[25]:


results = db.max_marginal_relevance_search(query, filter=filter)
for doc in results:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}\n\n")

# ## Q&A with Anthropic Claude and Retrieved Vectors

# Now that you are able to query from the vector store, you're ready to feed context into your LLM.
# 
# Using the LangChain wrapper for Bedrock, creating an object for the LLM can be done in a single line of code where you specify the model_id of the desired LLM (Claude V2 in this case), and any model level arguments.

# In[26]:


from langchain.llms.bedrock import Bedrock

model_kwargs_claude = { 
        "max_tokens_to_sample": 512,
        "stop_sequences": [],
        "temperature":0,  
        "top_p":0.5
    }

# Anthropic Claude Model
llm = Bedrock(
    model_id="anthropic.claude-v2", 
    client=bedrock_runtime, 
    model_kwargs=model_kwargs_claude,
    callbacks=[token_counter]
)

# Since you have a model object set up, you can use it to get a baseline of what the LLM will produce without any provided context.
# 
# Something you will notice is with the prompt "How has AWS evolved?", the answer isn't bad, but its not exactly what you'd look for from the lens of an executive. You'd want to hear about how they approached things that led to evolution, whereas the baseline results are just facts that indicate change. Later in the notebook, you will provide context to get a more tailored answer.

# In[27]:


print(llm.predict("How has AWS evolved?"))

token_counter.report()

# With your LLM ready to go, you'll create a prompt template to utilize context to answer a given question. Prompt formats will be different by model, so if you change your model you will also likely need to adjust your prompt.

# In[28]:


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt_template = """

Human: Here is a set of context, contained in <context> tags:

<context>
{context}
</context>

Use the context to provide an answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

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

# In[29]:


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "filter": filter},
        callbacks=[token_counter]
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
    callbacks=[token_counter]
)

# Now that your chain is set up, you can supply queries to it and generate responses based on your source documents.
# 
# You'll note that the LLM response references the context documents provided, using them to formulate a response calling out things that were mentioned specifically by Amazon's CEO.

# In[30]:


query = "How has AWS evolved?"
result = qa({"query": query})

print(f'Query: {result["query"]}\n')
print(f'Result: {result["result"]}\n')
print(f'Context Documents: ')
for srcdoc in result["source_documents"]:
      print(f'{srcdoc}\n')

# In[31]:


query = "Why is Amazon successful?"
result = qa({"query": query})

print(f'Query: {result["query"]}\n')
print(f'Result: {result["result"]}\n')
print(f'Context Documents: ')
for srcdoc in result["source_documents"]:
      print(f'{srcdoc}\n')

# In[32]:


query = "What business challenges has Amazon experienced?"
result = qa({"query": query})

print(f'Query: {result["query"]}\n')
print(f'Result: {result["result"]}\n')
print(f'Context Documents: ')
for srcdoc in result["source_documents"]:
      print(f'{srcdoc}\n')

# In[33]:


query = "How was Amazon impacted by COVID-19?"
result = qa({"query": query})

print(f'Query: {result["query"]}\n')
print(f'Result: {result["result"]}\n')
print(f'Context Documents: ')
for srcdoc in result["source_documents"]:
      print(f'{srcdoc}\n')

# In[ ]:



