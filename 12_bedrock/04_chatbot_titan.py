#!/usr/bin/env python
# coding: utf-8

# # Conversational Interface - Chatbot with Titan LLM
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*
# 
# In this notebook, we will build a chatbot using the Foundation Models (FMs) in Amazon Bedrock. For our use-case we use Titan as our FM for building the chatbot.

# ## Overview
# 
# Conversational interfaces such as chatbots and virtual assistants can be used to enhance the user experience for your customers. Chatbots uses natural language processing (NLP) and machine learning algorithms to understand and respond to user queries. Chatbots can be used in a variety of applications, such as customer service, sales, and e-commerce, to provide quick and efficient responses to users. They can be accessed through various channels such as websites, social media platforms, and messaging apps.
# 
# 
# ## Chatbot using Amazon Bedrock
# 
# ![Amazon Bedrock - Conversational Interface](./images/chatbot_bedrock.png)
# 
# 
# ## Use Cases
# 
# 1. **Chatbot (Basic)** - Zero Shot chatbot with a FM model
# 2. **Chatbot using prompt** - template(Langchain) - Chatbot with some context provided in the prompt template
# 3. **Chatbot with persona** - Chatbot with defined roles. i.e. Career Coach and Human interactions
# 4. **Contextual-aware chatbot** - Passing in context through an external file by generating embeddings
# 
# ## Langchain framework for building Chatbot with Amazon Bedrock
# In Conversational interfaces such as chatbots, it is highly important to remember previous interactions, both at a short term but also at a long term level.
# 
# LangChain provides memory components in two forms. First, LangChain provides helper utilities for managing and manipulating previous chat messages. These are designed to be modular and useful regardless of how they are used. Secondly, LangChain provides easy ways to incorporate these utilities into chains.
# It allows us to easily define and interact with different types of abstractions, which make it easy to build powerful chatbots.
# 
# ## Building Chatbot with Context - Key Elements
# 
# The first process in a building a contextual-aware chatbot is to **generate embeddings** for the context. Typically, you will have an ingestion process which will run through your embedding model and generate the embeddings which will be stored in a sort of a vector store. In this example we are using a Titan embeddings model for this.
# 
# ![Embeddings](./images/embeddings_lang.png)
# 
# Second process is the user request orchestration , interaction,  invoking and returing the results.
# 
# ![Chatbot](./images/chatbot_lang.png)
# 
# ## Architecture [Context Aware Chatbot]
# ![4](./images/context-aware-chatbot.png)

# ## Setup
# 
# Before running the rest of this notebook, you'll need to run the cells below to (ensure necessary libraries are installed and) connect to Bedrock.
# 
# For more details on how the setup works and ⚠️ **whether you might need to make any changes**, refer to the [Bedrock boto3 setup notebook](../00_Intro/bedrock_boto3_setup.ipynb) notebook.

# In[2]:


# %pip install boto3 botocore --force-reinstall --quiet
# %pip install langchain==0.0.309 --quiet 

# In[3]:


#### Un comment the following lines to run from your local environment outside of the AWS account with Bedrock access

#import os
#os.environ['BEDROCK_ASSUME_ROLE'] = '<YOUR_VALUES>'
#os.environ['AWS_PROFILE'] = '<YOUR_VALUES>'

# In[4]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# In this notebook, we'll also need some extra dependencies:
# 
# - [FAISS](https://github.com/facebookresearch/faiss), to store vector embeddings
# - [IPyWidgets](https://ipywidgets.readthedocs.io/en/stable/), for interactive UI widgets in the notebook
# - [PyPDF](https://pypi.org/project/pypdf/), for handling PDF files

# In[5]:


%pip install --quiet "faiss-cpu>=1.7,<2" "ipywidgets>=7,<8" langchain==0.0.309 "pypdf>=3.8,<4"

# ## Chatbot (Basic - without context)
# 
# #### Using CoversationChain from LangChain to start the conversation
# 
# Chatbots needs to remember the previous interactions. Conversational memory allows us to do that. There are several ways that we can implement conversational memory. In the context of LangChain, they are all built on top of the ConversationChain.
# 
# Note: The model outputs are non-deterministic

# In[6]:


from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory

titan_llm = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock_runtime)
titan_llm.model_kwargs = {'temperature': 0.5, "maxTokenCount": 700}

memory = ConversationBufferMemory()
memory.human_prefix = "User"
memory.ai_prefix = "Bot"

conversation = ConversationChain(
    llm=titan_llm, verbose=True, memory=memory
)
conversation.prompt.template = """System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer. The assistant is talkative and provides lots of specific details from it's context.\n\nCurrent conversation:\n{history}\nUser: {input}\nBot:"""

print(conversation.predict(input="Hi there!"))

# #### New Questions
# 
# Model has responded with intial message, let's ask few questions

# In[7]:


print(conversation.predict(input="Give me a few tips on how to start a new garden."))

# #### Build on the questions
# 
# Let's ask a question without mentioning the word garden to see if model can understand previous conversation

# In[8]:


print(conversation.predict(input="Cool. Will that work with tomatoes?"))

# #### Finishing this conversation

# In[9]:


print(conversation.predict(input="That's all, thank you!"))

# ## Chatbot using prompt template (Langchain)

# PromptTemplate is responsible for the construction of this input. LangChain provides several classes and functions to make constructing and working with prompts easy. We will use the default [PromptTemplate](https://python.langchain.com/en/latest/modules/prompts/getting_started.html) here.

# In[10]:


from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate

chat_history = []

memory=ConversationBufferMemory()
memory.human_prefix = "User"
memory.ai_prefix = "Bot"

# turn verbose to true to see the full logs and documents
qa = ConversationChain(
    llm=titan_llm, 
    verbose=False, 
    memory=memory #memory_chain
)
qa.prompt.template = """System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer. The assistant is talkative and provides lots of specific details from it's context.\n\nCurrent conversation:\n{history}\nUser: {input}\nBot:"""

print(f"ChatBot:DEFAULT:PROMPT:TEMPLATE: is ={qa.prompt.template}")

# In[11]:


import ipywidgets as ipw
from IPython.display import display, clear_output

class ChatUX:
    """ A chat UX using IPWidgets
    """
    def __init__(self, qa, retrievalChain = False):
        self.qa = qa
        self.name = None
        self.b=None
        self.retrievalChain = retrievalChain
        self.out = ipw.Output()


    def start_chat(self):
        print("Starting chat bot")
        display(self.out)
        self.chat(None)


    def chat(self, _):
        if self.name is None:
            prompt = ""
        else: 
            prompt = self.name.value
        if 'q' == prompt or 'quit' == prompt or 'Q' == prompt:
            print("Thank you , that was a nice chat !!")
            return
        elif len(prompt) > 0:
            with self.out:
                thinking = ipw.Label(value="Thinking...")
                display(thinking)
                try:
                    if self.retrievalChain:
                        result = self.qa.run({'question': prompt })
                    else:
                        result = self.qa.run({'input': prompt }) #, 'history':chat_history})
                except:
                    result = "No answer"
                thinking.value=""
                print(f"AI:{result}")
                self.name.disabled = True
                self.b.disabled = True
                self.name = None

        if self.name is None:
            with self.out:
                self.name = ipw.Text(description="You:", placeholder='q to quit')
                self.b = ipw.Button(description="Send")
                self.b.on_click(self.chat)
                display(ipw.Box(children=(self.name, self.b)))

# Let's start a chat

# In[12]:


chat = ChatUX(qa)
chat.start_chat()

# ## Chatbot with persona

# AI assistant will play the role of a career coach. Role Play Dialogue requires user message to be set in before starting the chat. ConversationBufferMemory is used to pre-populate the dialog.

# In[13]:


memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("You will be acting as a career coach. Your goal is to give career advice to users")
memory.chat_memory.add_ai_message("I am career coach and give career advice")

titan_llm = Bedrock(model_id="amazon.titan-text-express-v1",
                    client=bedrock_runtime)

conversation = ConversationChain(
     llm=titan_llm, verbose=True, memory=memory
)

print(conversation.predict(input="What are the career options in AI?"))

# ##### Let's ask a question that is not specaility of this Persona and the model shouldnn't answer that question and give a reason for that.

# In[14]:


conversation.verbose = False
print(conversation.predict(input="How to fix my car?"))

# ## Chatbot with Context 
# In this use case we will ask the Chatbot to answer question from the context that it was passed. We will take a csv file and use Titan embeddings Model to create the vector. This vector is stored in FAISS. When chatbot is asked a question we pass this vector and retrieve the answer. 

# #### Titan embeddings Model
# 
# Embeddings are a way to represent words, phrases or any other discrete items as vectors in a continuous vector space. This allows machine learning models to perform mathematical operations on these representations and capture semantic relationships between them.
# 
# 
# This will be used for the RAG [document search capability](https://labelbox.com/blog/how-vector-similarity-search-works/). 
# 

# In[15]:


from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate

br_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", 
                                  client=bedrock_runtime)

# #### Create the embeddings for document search

# #### FAISS as VectorStore
# 
# In order to be able to use embeddings for search, we need a store that can efficiently perform vector similarity searches. In this notebook we use FAISS, which is an in memory store. For permanently store vectors, one can use pgVector, Pinecone or Chroma.
# 
# The langchain VectorStore API's are available [here](https://python.langchain.com/en/harrison-docs-refactor-3-24/reference/modules/vectorstore.html)
# 
# To know more about the FAISS vector store please refer to this [document](https://arxiv.org/pdf/1702.08734.pdf).

# In[16]:


from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

s3_path = f"s3://jumpstart-cache-prod-us-east-2/training-datasets/Amazon_SageMaker_FAQs/Amazon_SageMaker_FAQs.csv"
!aws s3 cp $s3_path ./rag_data/Amazon_SageMaker_FAQs.csv

loader = CSVLoader("./rag_data/Amazon_SageMaker_FAQs.csv") # --- > 219 docs with 400 chars
documents_aws = loader.load() #
print(f"documents:loaded:size={len(documents_aws)}")

docs = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separator=",").split_documents(documents_aws)

print(f"Documents:after split and chunking size={len(docs)}")

vectorstore_faiss_aws = FAISS.from_documents(
    documents=docs,
    embedding = br_embeddings, 
    #**k_args
)

print(f"vectorstore_faiss_aws:created={vectorstore_faiss_aws}::")


# #### To run a quick low code test 
# 
# We can use a Wrapper class provided by LangChain to query the vector data base store and return to us the relevant documents. Behind the scenes this is only going to run a QA Chain with all default values

# In[17]:


wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss_aws)
print(wrapper_store_faiss.query("R in SageMaker", llm=titan_llm))

# #### Chatbot application
# 
# For the chatbot we need context management, history, vector stores, and many other things. We will start by with a ConversationalRetrievalChain
# 
# This uses conversation memory and RetrievalQAChain which Allow for passing in chat history which can be used for follow up questions.Source: https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html
# 
# Set verbose to True to see all the what is going on behind the scenes

# In[18]:


from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT


def create_prompt_template():
    _template = """{chat_history}

Answer only with the new question.
How would you ask the question considering the previous conversation: {question}
Question:"""
    CONVO_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    return CONVO_QUESTION_PROMPT

memory_chain = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
chat_history=[]

# #### Parameters used for ConversationRetrievalChain
# * **retriever**: We used `VectorStoreRetriever`, which is backed by a `VectorStore`. To retrieve text, there are two search types you can choose: `"similarity"` or `"mmr"`. `search_type="similarity"` uses similarity search in the retriever object where it selects text chunk vectors that are most similar to the question vector.
# 
# * **memory**: Memory Chain to store the history 
# 
# * **condense_question_prompt**: Given a question from the user, we use the previous conversation and that question to make up a standalone question
# 
# * **chain_type**: If the chat history is long and doesn't fit the context you use this parameter and the options are `stuff`, `refine`, `map_reduce`, `map-rerank`
# 
# If the question asked is outside the scope of context, then the model will reply it doesn't know the answer
# 
# **Note**: if you are curious how the chain works, uncomment the `verbose=True` line.

# In[19]:


# turn verbose to true to see the full logs and documents
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(
    llm=titan_llm, 
    retriever=vectorstore_faiss_aws.as_retriever(), 
    #retriever=vectorstore_faiss_aws.as_retriever(search_type='similarity', search_kwargs={"k": 8}),
    memory=memory_chain,
    #verbose=True,
    #condense_question_prompt=CONDENSE_QUESTION_PROMPT, # create_prompt_template(), 
    chain_type='stuff', # 'refine',
    #max_tokens_limit=100
)

qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template("""
{context}

Use at maximum 3 sentences to answer the question inside the <q></q> XML tags. 

<q>{question}</q>

Do not use any XML tags in the answer. If the answer is not in the context say "Sorry, I don't know, as the answer was not found in the context."

Answer:""")

# Let's start a chat

# In[ ]:


chat = ChatUX(qa, retrievalChain=True)
chat.start_chat()
