#!/usr/bin/env python
# coding: utf-8

# # ReAct Prompting with Langchain Agents with Claude 
# Reused from the [langchain handbook](https://github.com/pinecone-io/examples/tree/master/generation/langchain/handbook) 

# In[18]:


#!pip install -qU langchain sqlalchemy

# To run this notebook, we will need to use an OpenAI LLM. Here we will setup the LLM we will use for the whole notebook, just input your openai api key when prompted. 

# In[31]:


import os
import boto3
from pprint import pprint
boto3_bedrock = None #boto3.client('bedrock')


from langchain.llms.bedrock import Bedrock
titan_llm = Bedrock(model_id="amazon.titan-tg1-large", client=boto3_bedrock)
antropic_llm = Bedrock(model_id="anthropic.claude-v1", client=boto3_bedrock)

# ## What is an agent?

# **Definition**: The key behind agents is giving LLM's the possibility of using tools in their workflow. This is where langchain departs from the popular chatgpt implementation and we can start to get a glimpse of what it offers us as builders. Until now, we covered several building blocks in isolation. Let's see them come to life.
# 
# The official definition of agents is the following:
# 
# 
# > Agents use an LLM to determine which actions to take and in what order. An action can either be using a tool and observing its output, or returning to the user.

# In this edition we will cover what we may call 'generic' agents which really able to perform many meta tasks. There are other more specific agents that are tuned for different tasks (called 'agent-toolkits'), but we will cover those in a future edition.

# ## Create database

# We will use the agents to interact with a small sample database of stocks. We will not dive into the details because this is just a dummy tool we will build for illustrative purposes. Let's create it. 

# In[32]:


from sqlalchemy import MetaData

metadata_obj = MetaData()

# In[33]:


from sqlalchemy import Column, Integer, String, Table, Date, Float

hotels = Table(
    "hotels",
    metadata_obj,
	Column("hotel_id", Integer, primary_key=True),
	Column("city", String(15), nullable=False),
	Column("distance", Float, nullable=True),
	Column("stars", Float, nullable=True),
	Column("rating", Float, nullable=True),
	Column("country", String(15), nullable=True),
	Column("city_actual", String(15), nullable=True),
	Column("rating_reviewcount", Integer, nullable=True),
	Column("center1label", String(20), nullable=True),
	Column("center2label", String(20), nullable=True),
	Column("neighbourhood", String(15), nullable=True),
	Column("ratingta", Float, nullable=True),
	Column("ratingta_count", Integer, nullable=True),
	Column("distance_alter", Float, nullable=True),
	Column("accommodation_type", String(15), nullable=True),
    Column("timestamp", Date, nullable=True)
)

# In[34]:


from sqlalchemy import create_engine

engine = create_engine("sqlite:///:memory:")
metadata_obj.create_all(engine)

# In[79]:


import pandas as pd
from datetime import datetime

hotel_data = pd.read_csv("hotels-europe_features.csv")
hotel_data["timestamp"] = datetime(2023, 6, 28)
hotel_data

# In[36]:


observations = hotel_data

# In[37]:


observations = hotel_data.values.tolist()
print(observations[0])

# In[38]:


from sqlalchemy import insert

def insert_obs(obs):
    stmt = insert(hotels).values(
    hotel_id=obs[0], 
    city=obs[1], 
    distance=obs[2],
    stars=obs[3],
    rating=obs[4],
    country=obs[5],
    city_actual=obs[6],
    rating_reviewcount=obs[7],
    center1label=obs[8],
    center2label=obs[9],
    neighbourhood=obs[10],
    ratingta=obs[11],
    ratingta_count=obs[12],
    distance_alter=obs[13],
    accommodation_type=obs[14],
    timestamp=obs[15]
    )

    with engine.begin() as conn:
        conn.execute(stmt)

# In[39]:


for obs in observations:
    insert_obs(obs)

# In[40]:


from langchain.sql_database import SQLDatabase
from langchain.chains import SQLDatabaseChain

db = SQLDatabase(engine)
sql_chain = SQLDatabaseChain(llm=antropic_llm, database=db, verbose=True)

# Finally, we will create a tool with this chain. To create a custom tool we need only three inputs:
# * name: the name of the tool (sent to the llm as context)
# * func: the function that will be applied on the LLM's request
# * description: the description of the tool, what it should be used for (sent to the llm as context)

# In[41]:


from langchain.agents import Tool

sql_tool = Tool(
    name='Hotels DB',
    func=sql_chain.run,
    description="Useful for when you need to answer questions about hotels " \
                "and their ratings."
    
)

# ## Agent types

# In this section we will review several agents and see how they 'think' and what they can do.

# Using one of langchain's pre-built agents involves three variables: 
# * defining the tools
# * defining the llm
# * defining the agent type
# 
# This is all really easy to do in langchain, as we will see in the following example.

# ### Agent type #1: Zero Shot React

# We will be using only a few tools but you can combine the ones you prefer. Remember that tools are only utility chains under the hood which are chains that serve one specific purpose.

# We first need to initialize the tools we'll be using in this example. Here we will want our agent to do math so we will use 'llm-math', the tool we have for solving math problems. This is one of the several tools we can load with the `load_tools` utility - you can check out more by checking out the [docs](https://langchain.readthedocs.io/en/latest/modules/agents/tools.html?highlight=tools#tools).

# In[42]:


llm = antropic_llm

# In[43]:


from langchain.agents import load_tools

tools = load_tools(
    ["llm-math"], 
    llm=llm
)


# In[44]:


tools.append(sql_tool)

# In[45]:


tools

# To be able to search across our stock prices, we will also use the custom tool we built beforehand with the sql data ('Stock DB'). We will append this tool to our list of tools.

# As the name suggests, we will use this agent to perform 'zero shot' tasks on the input. That means that we will not have several, interdependent interactions but only one. In other words, this agent will have no memory.

# Now we are ready to initialize the agent! We will use `verbose` in `True` so we can see what is our agent's 'thinking' process.

# **Important Note:** *When interacting with agents it is really important to set the `max_iterations` parameters because agents can get stuck in infinite loops that consume plenty of tokens. The default value is 15 to allow for many tools and complex reasoning but for most applications you should keep it much lower.*

# In[46]:


from langchain.agents import initialize_agent

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description", 
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=1,
)

# Let's see our newly created agent in action! We will ask it a question that involves a math operation over the stock prices.

# In[73]:


hotels_query="""Search for 3 hotels in Madrid with 4 stars and highest rating"""

# In[75]:


result = zero_shot_agent(hotels_query)

# In[76]:


print(zero_shot_agent.agent.llm_chain.prompt.template)

# The question we must ask ourselves here is: how are agents different than chains?

# If we look at the agent's logic and the prompt we have just printed we will see some clear differences. First, we have the tools which are included in the prompt. Second we have a thought process which was before was immediate in chains but now involves a 'thought', 'action', 'action input', 'observation' sequence. What is this all about?

# Suffice it to say for now that **the LLM now has the ability to 'reason' on how to best use tools** to solve our query and can combine them in intelligent ways with just a brief description of each of them. If you want to learn more about this paradigm (MRKL) in detail, please refer to [this](https://arxiv.org/pdf/2205.00445.pdf) paper. 

# Finally, let's pay attention to the 'agent_scratchpad'. What is that? Well, that is where we will be appending every thought or action that the agent has already performed. In this way, at each point in time, the agent will know what it has found out and will be able to continue its thought process. In other words, after using a tool it adds its thoughts and observations to the scratchpad and picks up from there.

# ### Agent type #2: Conversational React

# The zero shot agent is really interesting but, as we said before, it has no memory. What if we want an assistant that remembers things we have talked about and can also reason about them and use tools? For that we have the conversational react agent.

# We will use the same tools as we specified earlier:

# In[61]:


tools = load_tools(
    ["llm-math"], 
    llm=llm
)

# In[62]:


tools.append(sql_tool)

# The memory type being used here is a simple buffer memory to allow us to remember previous steps in the reasoning chain. For more information on memory, please refer to the 3rd chapter of this series.

# In[63]:


from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

# In[64]:


conversational_agent = initialize_agent(
    agent='conversational-react-description', 
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=1,
    memory=memory,
)

# In[78]:


result = conversational_agent("Please provide me a hotel with 4 stars near Madrid")

# As we can see below, the prompt is similar but it includes a great prelude of instructions that make it an effective assistant as well + a spot for including the chat history from the memory component:
