#!/usr/bin/env python
# coding: utf-8

# # AI Assistant

# In this notebook you'll make an AI assistant to help customers make the best decision about getting a new shoe from Cloudrunners Shoes. You will also take a short dive into practical token limitations for the model you are working with, and where you can focus your learning in the case you need to build more powerful AI agent systems.

# ## Helper Functions

# ### Amazon SageMaker
# Use Amazon SageMaker Jumpstart to deploy a Llama2-70B model. 

# ## Setup
# 
# ***

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

# In[5]:


from sagemaker.jumpstart.model import JumpStartModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = JumpStartModel(model_id=model_id)
predictor = model.deploy()

# Load the model and tokenizer
model_name = "meta-llama/LLaMA-3-7b"  # Replace with your local model name
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def local_model_inference(prompt, temperature=0.6):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=512, top_p=0.9, temperature=temperature)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Replace the generate function in your script with local_model_inference
def generate(prompt, temperature=0.6):
    return local_model_inference(prompt, temperature)

# In[6]:


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

# ### Edit System Message

# In[7]:


def prompt_with_system_message(prompt, system_message):
    prompt = f"""
    <s>[INST] <<SYS>>{system_message}<</SYS>>

    User: {prompt}
    Agent:[/INST]
    """
    return prompt

# ### Include One-to-many Shot Learning

# In[8]:


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

# In[9]:


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

# ## Data

# ### Cloudrunners Shoes Details

# In[10]:


shoes = [
    {
        "model": "Sky Glider",
        "type": "Running", 
        "features": {
            "upper": "Mesh",
            "sole": "EVA foam",
            "lacing": "Quick-tie",
            "drop": "8mm",
            "color": "Blue with yellow accents"
        },
        "usps": ["Lightweight", "Responsive cushioning", "Seamless upper"],
        "price": 119.95,
        "internal_id": "SG123",
        "weight": "220g",
        "manufacturer_location": "Vietnam"
    },
    {   
        "model": "Trail Trekker",
        "type": "Hiking",
        "features": {
            "upper": "Synthetic leather",
            "sole": "Rubber lug", 
            "lacing": "Traditional",
            "drop": "12mm",
            "color": "Khaki green"
        },
        "usps": ["Rugged construction", "Super grippy sole", "Waterproof"], 
        "price": 129.99,
        "internal_id": "TT321",
        "weight": "340g",
        "manufacturer_location": "China"
    },
    {
        "model": "Marathon Master",
        "type": "Racing",
        "features": {
            "upper": "Mesh",
            "sole": "Carbon fiber plate",
            "lacing": "Speed laces",
            "drop": "6mm",
            "color": "Neon yellow and black"
        },
        "usps": ["Maximizes energy return", "Lightning fast", "Seamless comfort"],
        "price": 179.50, 
        "internal_id": "MM111",
        "weight": "180g",
        "manufacturer_location": "USA"
    }
]

# ## Shoes AI Assistant

# In[11]:


system_message = """
You are a friendly chatbot knowledgeable about shoes. \
When asked about specific shoe models or features, you try to provide accurate and helpful answers. \
Your goal is to assist and inform potential customers to the best of your ability.
"""

chatbot = LlamaChatbot(system_message)

# In[12]:


print(chatbot.chat("Can you tell me about the latest models?"))

# In[13]:


chatbot.reset()

# ## Cloudrunners Shoes AI Assistant

# In[14]:


system_message = f"""
You are a friendly chatbot knowledgeable about these bicycles from Cloudrunners Shoes {shoes}. \
When asked about specific shoe models or features, you try to provide accurate and helpful answers. \
Your goal is to assist and inform potential customers to the best of your ability.
"""

chatbot = LlamaChatbot(system_message)

# In[15]:


print(chatbot.chat("Can you tell me about the latest models?"))

# In[16]:


print(chatbot.chat("How much do each of the models cost?"))

# In[17]:


print(chatbot.chat("I'm torn between the Sky Glider and the Marathon Master, how should I decide which is best or me?"))

# In[18]:


print(chatbot.chat("I'm torn between the Marathon Master and the Trail Trekker, how should I decide which is best or me?"))

# In[19]:


chatbot.reset()

# In[ ]:



