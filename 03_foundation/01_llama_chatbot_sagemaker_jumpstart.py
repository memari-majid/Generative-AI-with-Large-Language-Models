#!/usr/bin/env python
# coding: utf-8

# # Chatbot

# In this notebook you will begin to create chatbot functionality, creating an AI bot capable of retaining conversation history.

# ## Helper Functions

# ### Amazon SageMaker
# Use Amazon SageMaker Jumpstart to deploy a Llama2-70B model. 

# In[2]:


# %pip install --upgrade --quiet sagemaker

# ***
# You can continue with the default model or choose a different model: this notebook will run with the following model IDs :
# - `meta-textgeneration-llama-2-7b-f`
# - `meta-textgeneration-llama-2-13b-f`
# - `meta-textgeneration-llama-2-70b-f`
# ***

# In[3]:


model_id, model_version = "meta-textgeneration-llama-2-70b-f", "2.*"

# ## Deploy model
# 
# ***
# You can now deploy the model using SageMaker JumpStart.
# ***

# In[4]:


from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(model_id=model_id,
                       model_version=model_version)
predictor = model.deploy()

# In[5]:


endpoint_name = predictor.endpoint_name

import boto3, json
sm_client = boto3.client('sagemaker-runtime')

def transform_input(prompt: str, temperature) -> bytes:
    input_str = json.dumps({
        "inputs": [[{"role": "user", "content": prompt},]],
        "parameters": {"max_new_tokens": 512,"top_p": 0.9,"temperature": temperature}
    })
    return input_str.encode('utf-8')
    
def transform_output(output: bytes) -> str:
    response_json = json.loads(output['Body'].read().decode("utf-8"))
    return response_json[0]["generation"]["content"]

def generate(prompt, temperature = 0.6):
    response = sm_client.invoke_endpoint(
        EndpointName=endpoint_name, 
        CustomAttributes="accept_eula=true", 
        ContentType="application/json",
        Body=transform_input(prompt, temperature)
    )
    response_output = transform_output(response)
    return response_output

# ### Edit System Message

# In[6]:


def prompt_with_system_message(prompt, system_message):
    prompt = f"""
    <s>[INST] <<SYS>>{system_message}<</SYS>>

    User: {prompt}
    Agent:[/INST]
    """
    return prompt

# ### Include One-to-many Shot Learning

# In[7]:


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

# In[8]:


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

# In[9]:


prompt = "Hello my name is Rock. Nice to meet you!"

print(generate(prompt))

# In[10]:


prompt = "Can you remind me what my name is?"

print(generate(prompt))

# In[11]:


system_message = """
You are a friendly chatbot always eager to help and engage in meaningful conversation. \
You always remember the details of our previous conversations, \
especially if a user gives them their name.
"""

prompt = "Hello my name is Rock. Nice to meet you!"

print(generate(prompt_with_system_message(prompt, system_message)))

# In[12]:


system_message = """
You are a friendly chatbot always eager to help and engage in meaningful conversation. \
You always remember the details of our previous conversations, \
especially if a user gives them their name.
"""

prompt = "Can you remind me what my name is?"

print(generate(prompt_with_system_message(prompt, system_message)))

# ## Create Conversation Memory

# In[13]:


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

# In[14]:


system_message = """
You are a friendly chatbot always eager to help and engage in meaningful conversation.
"""

chatbot = LlamaChatbot(system_message)

# In[15]:


print(chatbot.chat("Hi, my name is Rock. Nice to meet you!"))

# In[16]:


print(chatbot.chat("Can you remind me what my name is?"))

# In[17]:


chatbot.reset()

# In[18]:


print(chatbot.chat("Can you remind me what my name is?"))

# In[ ]:



