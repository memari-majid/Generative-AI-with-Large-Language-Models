#!/usr/bin/env python
# coding: utf-8

# # Run Llama 2 Models in SageMaker JumpStart

# In[5]:


model_id = "meta-textgeneration-llama-2-7b-f"

# In[9]:


model_version = "2.*"

# ## Deploy model
# 
# ***
# You can now deploy the model using SageMaker JumpStart. For successful deployment, you must manually change the `accept_eula` argument in the model's deploy method to `True`.
# ***

# In[10]:


from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(model_id=model_id, model_version=model_version)

# In[11]:


predictor = model.deploy(accept_eula=True)

# ## Invoke the endpoint
# 
# ***
# ### Supported Parameters
# 
# ***
# This model supports many parameters while performing inference. They include:
# 
# * **max_length:** Model generates text until the output length (which includes the input context length) reaches `max_length`. If specified, it must be a positive integer.
# * **max_new_tokens:** Model generates text until the output length (excluding the input context length) reaches `max_new_tokens`. If specified, it must be a positive integer.
# * **num_beams:** Number of beams used in the greedy search. If specified, it must be integer greater than or equal to `num_return_sequences`.
# * **no_repeat_ngram_size:** Model ensures that a sequence of words of `no_repeat_ngram_size` is not repeated in the output sequence. If specified, it must be a positive integer greater than 1.
# * **temperature:** Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If `temperature` -> 0, it results in greedy decoding. If specified, it must be a positive float.
# * **early_stopping:** If True, text generation is finished when all beam hypotheses reach the end of sentence token. If specified, it must be boolean.
# * **do_sample:** If True, sample the next word as per the likelihood. If specified, it must be boolean.
# * **top_k:** In each step of text generation, sample from only the `top_k` most likely words. If specified, it must be a positive integer.
# * **top_p:** In each step of text generation, sample from the smallest possible set of words with cumulative probability `top_p`. If specified, it must be a float between 0 and 1.
# * **return_full_text:** If True, input text will be part of the output generated text. If specified, it must be boolean. The default value for it is False.
# * **stop**: If specified, it must be a list of strings. Text generation stops if any one of the specified strings is generated.
# 
# We may specify any subset of the parameters mentioned above while invoking an endpoint. Next, we show an example of how to invoke endpoint with these arguments.
# 
# **NOTE**: If `max_new_tokens` is not defined, the model may generate up to the maximum total tokens allowed, which is 4K for these models. This may result in endpoint query timeout errors, so it is recommended to set `max_new_tokens` when possible. For 7B, 13B, and 70B models, we recommend to set `max_new_tokens` no greater than 1500, 1000, and 500 respectively, while keeping the total number of tokens less than 4K.
# 
# **NOTE**: This model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and alternating (u/a/u/a/u...).
# 
# ***

# ### Example prompts
# ***
# The examples in this section demonstrate how to perform text generation with conversational dialog as prompt inputs. Example payloads are retrieved programmatically from the `JumpStartModel` object.
# 
# Input messages for Llama-2 chat models should exhibit the following format. The model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and alternating (u/a/u/a/u...). The last message must be from 'user'. A simple user prompt may look like the following:
# ```
# <s>[INST] {user_prompt} [/INST]
# ```
# You may also add a system prompt with the following syntax:
# ```
# <s>[INST] <<SYS>>
# {system_prompt}
# <</SYS>>
# 
# {user_prompt} [/INST]
# ```
# Finally, you can have a conversational interaction with the model by including all previous user prompts and assistant responses in the input:
# ```
# <s>[INST] <<SYS>>
# {system_prompt}
# <</SYS>>
# 
# {user_prompt_1} [/INST] {assistant_response_1} </s><s>[INST] {user_prompt_1} [/INST]
# ```
# ***

# In[28]:


example_payloads = model.retrieve_all_examples()
example_payloads[0].body

# In[29]:


response = predictor.predict(example_payloads[0].body, custom_attributes="accept_eula=true")
    
print(response[0]["generation"]["content"])

# ***
# While not used in the previously provided example payloads, you can format your own messages to the Llama-2 model with the following utility function.
# ***

# In[30]:


payload = {
    'inputs': [
        [{
            'role': 'system',
            'content': 'Always answer with Haiku.'
         },
         {
            'role': 'user',
            'content': 'I am going to Paris, what should I see?'
         }]            
    ],
    'parameters': {
        'max_new_tokens': 512, 'top_p': 0.9, 'temperature': 0.6
    }
}

response = predictor.predict(payload, custom_attributes="accept_eula=true")

print(response[0]["generation"]["content"])

# ## Clean up the endpoint

# In[ ]:


# # Delete the SageMaker endpoint
# predictor.delete_model()
# predictor.delete_endpoint()
