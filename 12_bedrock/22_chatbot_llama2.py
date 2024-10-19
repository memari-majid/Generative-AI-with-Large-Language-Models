#!/usr/bin/env python
# coding: utf-8

# # Chatbot

# In this notebook you will begin to create chatbot functionality, creating an AI bot capable of retaining conversation history.

# ## Helper Functions
# Comment/uncomment the sections below depending on the AWS service you are using for invoking Llama2.

# ### Amazon Bedrock

# In[2]:


import boto3, json
br_client = boto3.client('bedrock-runtime')

def generate(prompt, temperature = 0):
    body = json.dumps({
        "prompt": prompt,
        "temperature": temperature,
        "top_p": 0.9,
        "max_gen_len":512
    })
    response = br_client.invoke_model(body=body, modelId='meta.llama2-13b-chat-v1')
    response = json.loads(response.get('body').read())
    response = response.get('generation')
    return response

# ### Edit System Message

# In[3]:


def prompt_with_system_message(prompt, system_message):
    prompt = f"""
    <s>[INST] <<SYS>>{system_message}<</SYS>>

    User: {prompt}
    Agent:[/INST]
    """
    return prompt

# ### Include One-to-many Shot Learning

# In[4]:


def prompt_with_examples(prompt, system_message, examples=[]):
    
    # Start with the initial part of the prompt with system message
    full_prompt = f"<s>[INST] <<SYS>>{system_message}<</SYS>>\n"

    # Add each example to the prompt
    for user_msg, agent_response in examples:
        full_prompt += f"{user_msg} [/INST] {agent_response} </s><s>[INST]"

    # Add the main prompt and close the template
    full_prompt += f"{prompt} [/INST]"

    return full_prompt

# ### LlamaChatbot Class

# In[5]:


class LlamaChatbot:
    def __init__(self, system_message):
        self.system_message = system_message
        self.conversation_history = []  # list of tuples (user_msg, agent_response)

    def chat(self, user_msg):
        # Generate the prompt using the conversation history and the new user message
        prompt = prompt_with_examples(user_msg, self.system_message, self.conversation_history)
        
        # Get the model's response
        agent_response = generate(prompt)

        # Store this interaction in the conversation history
        self.conversation_history.append((user_msg, agent_response))

        return agent_response

    def reset(self):
        # Clear conversation history
        self.conversation_history = []

# ## No Conversation Memory

# In[6]:


prompt = "Hello my name is Rock. Nice to meet you!"

print(generate(prompt))

# In[7]:


prompt = "Can you remind me what my name is?"

print(generate(prompt))

# In[8]:


system_message = """
You are a friendly chatbot always eager to help and engage in meaningful conversation. \
You always remember the details of our previous conversations, \
especially if a user gives them their name.
"""

prompt = "Hello my name is Rock. Nice to meet you!"

print(generate(prompt_with_system_message(prompt, system_message)))

# In[9]:


system_message = """
You are a friendly chatbot always eager to help and engage in meaningful conversation. \
You always remember the details of our previous conversations, \
especially if a user gives them their name.
"""

prompt = "Can you remind me what my name is?"

print(generate(prompt_with_system_message(prompt, system_message)))

# ## Create Conversation Memory

# In[10]:


class LlamaChatbot:
    def __init__(self, system_message):
        self.system_message = system_message
        self.conversation_history = []  # list of tuples (user_msg, agent_response)

    def chat(self, user_msg):
        # Generate the prompt using the conversation history and the new user message
        prompt = prompt_with_examples(user_msg, self.system_message, self.conversation_history)
        
        # Get the model's response
        agent_response = generate(prompt)

        # Store this interaction in the conversation history
        self.conversation_history.append((user_msg, agent_response))

        return agent_response

    def reset(self):
        # Clear conversation history
        self.conversation_history = []

# In[11]:


system_message = """
You are a friendly chatbot always eager to help and engage in meaningful conversation.
"""

chatbot = LlamaChatbot(system_message)

# In[12]:


print(chatbot.chat("Hi, my name is Rock. Nice to meet you!"))

# In[13]:


print(chatbot.chat("Can you remind me what my name is?"))

# In[14]:


chatbot.reset()

# In[15]:


print(chatbot.chat("Can you remind me what my name is?"))
