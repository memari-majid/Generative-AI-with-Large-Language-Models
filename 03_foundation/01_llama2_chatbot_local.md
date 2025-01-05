#!/usr/bin/env python
# coding: utf-8

# # Chatbot

# This document provides instructions and code to create a chatbot capable of retaining conversation history using a local model.

# ## Setup Instructions

# 1. **Install Python Packages**: You need to install the `transformers` and `torch` libraries from Hugging Face. Use the following command:
#    ```bash
#    pip install transformers torch
#    ```

# 2. **Download a Model**: Use a pre-trained model from Hugging Face's model hub. For local use, a smaller model like `distilgpt2` is recommended. Here is how you can download and set up the model:
#    ```python
#    from transformers import AutoModelForCausalLM, AutoTokenizer

#    model_name = "distilgpt2"  # Smaller model for local use
#    model = AutoModelForCausalLM.from_pretrained(model_name)
#    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    ```

# ## Helper Functions

# ### Local Model Inference
# Use a local model inference method to deploy a smaller model. 

# In[2]:


from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model and tokenizer
model_name = "distilgpt2"  # Smaller model for local use
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a function to perform local model inference
def local_model_inference(prompt, temperature=0.6):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate a response using the model
    outputs = model.generate(**inputs, max_new_tokens=512, top_p=0.9, temperature=temperature)

    # Decode the generated tokens to a string
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        agent_response = local_model_inference(prompt)

        # Store this interaction in the conversation history
        self.conversation_history.append((user_msg, agent_response))

        return agent_response

    def reset(self):
        # Clear conversation history
        self.conversation_history = []

# ## Example Usage

# In[6]:


system_message = """
You are a friendly chatbot always eager to help and engage in meaningful conversation.
"""

chatbot = LlamaChatbot(system_message)

# In[7]:


print(chatbot.chat("Hi, my name is Rock. Nice to meet you!"))

# In[8]:


print(chatbot.chat("Can you remind me what my name is?"))

# In[9]:


chatbot.reset()

# In[10]:


print(chatbot.chat("Can you remind me what my name is?")) 