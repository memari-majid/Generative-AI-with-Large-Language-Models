#!/usr/bin/env python
# coding: utf-8

# # Run Llama 2 Models in SageMaker JumpStart

# ---
# In this demo notebook, we demonstrate how to use the SageMaker Python SDK to deploy a JumpStart model for Text Generation using the Llama 2 fine-tuned model optimized for dialogue use cases.
# 
# To perform inference on these models, you need to pass custom_attributes='accept_eula=true' as part of header. This means you have read and accept the end-user-license-agreement (EULA) of the model. EULA can be found in model card description or from https://ai.meta.com/resources/models-and-libraries/llama-downloads/. By default, this notebook sets custom_attributes='accept_eula=false', so all inference requests will fail until you explicitly change this custom attribute.
# 
# Note: Custom_attributes used to pass EULA are key/value pairs. The key and value are separated by '=' and pairs are separated by ';'. If the user passes the same key more than once, the last value is kept and passed to the script handler (i.e., in this case, used for conditional logic). For example, if 'accept_eula=false; accept_eula=true' is passed to the server, then 'accept_eula=true' is kept and passed to the script handler.
# 
# ---

# ## Setup
# 
# ***

# In[2]:


%pip install --upgrade --quiet sagemaker

# ***
# You can continue with the default model or choose a different model: this notebook will run with the following model IDs :
# - `meta-textgeneration-llama-2-7b-f`
# - `meta-textgeneration-llama-2-13b-f`
# - `meta-textgeneration-llama-2-70b-f`
# ***

# In[3]:


model_id, model_version = "meta-textgeneration-llama-2-70b-f", "*"

# ## Deploy model
# 
# ***
# You can now deploy the model using SageMaker JumpStart.
# ***

# In[4]:


from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(model_id=model_id)
predictor = model.deploy()

# ## Invoke the endpoint
# 
# ***
# ### Supported Parameters
# This model supports the following inference payload parameters:
# 
# * **max_new_tokens:** Model generates text until the output length (excluding the input context length) reaches max_new_tokens. If specified, it must be a positive integer.
# * **temperature:** Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If `temperature` -> 0, it results in greedy decoding. If specified, it must be a positive float.
# * **top_p:** In each step of text generation, sample from the smallest possible set of words with cumulative probability `top_p`. If specified, it must be a float between 0 and 1.
# 
# You may specify any subset of the parameters mentioned above while invoking an endpoint. 
# 
# ***
# ### Notes
# - If `max_new_tokens` is not defined, the model may generate up to the maximum total tokens allowed, which is 4K for these models. This may result in endpoint query timeout errors, so it is recommended to set `max_new_tokens` when possible. For 7B, 13B, and 70B models, we recommend to set `max_new_tokens` no greater than 1500, 1000, and 500 respectively, while keeping the total number of tokens less than 4K.
# - In order to support a 4k context length, this model has restricted query payloads to only utilize a batch size of 1. Payloads with larger batch sizes will receive an endpoint error prior to inference.
# - This model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and alternating (u/a/u/a/u...).
# 
# ***

# In[5]:


def print_dialog(payload, response):
    dialog = payload["inputs"][0]
    for msg in dialog:
        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    print(f"> {response[0]['generation']['role'].capitalize()}: {response[0]['generation']['content']}")
    print("\n==================================\n")

# ### Example 1

# In[18]:


%%time

payload = {
    "inputs": [[
        {"role": "user", "content": "what is the recipe of mayonnaise?"},
    ]],
    "parameters": {"max_new_tokens": 512, "top_p": 0.9, "temperature": 0.6}
}
response = predictor.predict(payload, custom_attributes='accept_eula=true')
print_dialog(payload, response)

# ### Example 2

# In[19]:


%%time

payload = {
    "inputs": [[
        {"role": "user", "content": "I am going to Paris, what should I see?"},
        {
            "role": "assistant",
            "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
        },
        {"role": "user", "content": "What is so great about #1?"},
    ]],
    "parameters": {"max_new_tokens": 512, "top_p": 0.9, "temperature": 0.6}
}
response = predictor.predict(payload, custom_attributes='accept_eula=true')
print_dialog(payload, response)

# ### Example 3

# In[20]:


%%time

payload = {
    "inputs": [[
        {"role": "system", "content": "Always answer with Haiku"},
        {"role": "user", "content": "I am going to Paris, what should I see?"},
    ]],
    "parameters": {"max_new_tokens": 512, "top_p": 0.9, "temperature": 0.6}
}
response = predictor.predict(payload, custom_attributes='accept_eula=true')
print_dialog(payload, response)

# ### Example 4

# In[21]:


%%time

payload = {
    "inputs": [[
        {"role": "system", "content": "Always answer with emojis"},
        {"role": "user", "content": "How to go from Beijing to NY?"},
    ]],
    "parameters": {"max_new_tokens": 512, "top_p": 0.9, "temperature": 0.6}
}
response = predictor.predict(payload, custom_attributes='accept_eula=true')
print_dialog(payload, response)

# ## Clean up the endpoint

# In[ ]:


# # Delete the SageMaker endpoint
# predictor.delete_model()
# predictor.delete_endpoint()
