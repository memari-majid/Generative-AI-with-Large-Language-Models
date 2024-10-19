#!/usr/bin/env python
# coding: utf-8

# # Bedrock Prompt Examples for Insurance
# 
# In this notebook, we include different example use cases for Insurance using Amazon Bedrock.
# 
# These sample use cases involve different tasks and prompt engeering techniques, as follows:
# 1. **Generate recommendations based on metadata**
#     - **Task:** Text Generation
#     - **Prompt Technique:** Zero-shot
# 2. **Estimate people insured for given policies based on historical data**
#     - **Task:** Complex Reasoning
#     - **Prompt Technique:** Chain-of-Thoughts (CoT)
# 3. **Create a question answering assistant for customer service**
#     - **Task:** Question Answering with Dialogue Asistant (without memory)
#     - **Prompt Technique:** Few-shot
# 4. **Summarize and classify content from media files transcription**
#     - **Task:** Text Summarization & Text Classification
#     - **Prompt Technique:** Zero-shot
# 5. **Create splash pages describing upcoming promotions**
#     - **Task:** Code Generation
#     - **Prompt Technique:** Zero-shot

# Let's start by ensuring the Bedrock SDK is properly installed.
# 
# We'll also install a few libraries required in the notebook.

# In[ ]:


#!unzip ../bedrock-preview-documentation/SDK/bedrock-python-sdk.zip -d /root/bedrock

#!pip install --upgrade pip
#!pip install scikit-learn seaborn

#!pwd
#!python3 -m pip install /root/bedrock/boto3-1.26.142-py3-none-any.whl
#!python3 -m pip install /root/bedrock/botocore-1.29.142-py3-none-any.whl

# Now we can import the libraries and setup the Bedrock client.

# In[12]:


import boto3
import json
import csv
from datetime import datetime

bedrock = boto3.client(
 service_name='bedrock',
 region_name='us-east-1',
 endpoint_url='https://bedrock.us-east-1.amazonaws.com'
)

# Let's get the list of Foundational Models supported in Bedrock at this time.

# In[13]:


bedrock.list_foundation_models()

# We will define an utility function for calling Bedrock.
# 
# This will help passing the proper body depending on the model invoked, and will store the results in a CSV file as well.

# In[14]:


def call_bedrock(modelId, prompt_data):
    if 'amazon' in modelId:
        body = json.dumps({
            "inputText": prompt_data,
            "textGenerationConfig":
            {
                "maxTokenCount":4096,
                "stopSequences":[],
                "temperature":0,
                "topP":0.9
            }
        })
        #modelId = 'amazon.titan-tg1-large'
    elif 'anthropic' in modelId:
        body = json.dumps({
            "prompt": prompt_data,
            "max_tokens_to_sample": 4096,
            "stop_sequences":[],
            "temperature":0,
            "top_p":0.9
        })
        #modelId = 'anthropic.claude-instant-v1'
    elif 'ai21' in modelId:
        body = json.dumps({
            "prompt": prompt_data,
            "maxTokens":4096,
            "stopSequences":[],
            "temperature":0,
            "topP":0.9
        })
        #modelId = 'ai21.j2-grande-instruct'
    elif 'stability' in modelId:
        body = json.dumps({"text_prompts":[{"text":prompt_data}]}) 
        #modelId = 'stability.stable-diffusion-xl'
    else:
        print('Parameter model must be one of titan, claude, j2, or sd')
        return
    accept = 'application/json'
    contentType = 'application/json'

    before = datetime.now()
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    latency = (datetime.now() - before)
    response_body = json.loads(response.get('body').read())

    if 'amazon' in modelId:
        response = response_body.get('results')[0].get('outputText')
    elif 'anthropic' in modelId:
        response = response_body.get('completion')
    elif 'ai21' in modelId:
        response = response_body.get('completions')[0].get('data').get('text')

    return response, latency

# Now we are ready for running our examples with different models.
# 
# -----

# ## 1. Generate recommendations based on metadata
# 
# **Use Case:** A company wants to generate recommendations of insurance policies for their users based on some metadata, e.g. country, age-range, and interests.
# 
# **Task:** Text Generation
# 
# **Prompt Technique:** Zero-shot

# In[15]:


prompt_data ="""
Human:
Generate a list of 3 recommended insurance policies for a customer considering the information in the <metadata></metadata> XML tags, and include a very brief description of each recommendation.

<metadata>
Lives in Spain
Age range between 20-30
Owns a house
Owns a car
</metadata>

Assistant:
"""

# In[16]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[17]:


response, latency = call_bedrock('anthropic.claude-instant-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------

# ## 2. Estimate capacity for airlines or hotel properties based on historical data
# 
# **Use Case:** An insurance company wants to estimate contracted policies they could have for the next days based on historical information and metadata.
# 
# **Task:** Complex Reasoning
# 
# **Prompt Technique:** Chain-of-Thoughts (CoT)

# In[24]:


prompt_data ="""
Human: La semana pasada, los asegurados de 3 productos de pólizas de seguro de una empresa fueron según los siguientes datos:
- Lunes: Vida 65000, Auto 32000, Hogar 41500
- Martes: Vida 64000, Auto 33000, Hogar 41000
- Miércoles: Vida 63000, Auto 34000, Hogar 42500

Pregunta: ¿Cuántos asegurados podemos esperar el próximo Viernes para la póliza de Vida?
Respuesta: Según los números dados y sin disponer de más información, hay una disminución diaria de 1000 asegurados para la póliza de Vida.
Si asumimos que esta tendencia continuará durante los siguientes días, podemos esperar 62000 asegurados en la póliza de Vida para el siguiente día que es Jueves y,
por lo tanto, 61000 asegurados para el Viernes.

Pregunta: ¿Cuántos asegurados podemos esperar el Sábado en cada una de los productos? razona paso a paso y provee recomendaciones para incrementar los asegurados
Assistant:
Respuesta:
"""

# In[25]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[26]:


response, latency = call_bedrock('anthropic.claude-instant-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------

# ## 3. Create a question answering assistant for customer service
# 
# **Use Case:** A company wants to create a bot capable of answering questions about the services available, based on the internal information for these.
# 
# **Task:** Question Answering with Dialogue Asistant (no memory)
# 
# **Prompt Technique:** Few-shot

# In[27]:


prompt_data ="""
Context: An insurance company offers the following packages for contracting a travel insurance policy:
1. International Plan, price is USD 10 per day, includes travel inconvenience, medical, presonal accident, and COVID protection 
2. Domestic Plan, price is USD 3 per day, includes medical without COVID, and personal accidents

Instruction: Answer any questions about the services available in a friendly manner. If you don't know the answer just say 'Apologies, but I don't have the answer for that. Please contact our team by phone.'

Assistant: Welcome to Smily Insurance, how can I help you?
Human: Hi, I would like to know what are the travel insurance options available please.
Assistant: Of course, right now we have the Domestic Plan and the International Plan.
Human: Thank you. I would like to know details for those please.
Assistant:
"""

# In[28]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[29]:


response, latency = call_bedrock('anthropic.claude-instant-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------

# ## 4. Generate content summary based on transcriptions from media files
# 
# **Use Case:** A company needs to generate summaries of the audio transcription for audio and video files for customer service, to be sent to their operations quality team. They also want to classify this content according to specific categories.
# 
# **Task:** Text Summarization & Text Classification
# 
# **Prompt Technique:** Zero-shot

# #### (Pre-requisite)
# 
# First, we will need to transcribe the media files. You can e.g. use Amazon Transcribe for this task following examples like this: https://docs.aws.amazon.com/code-library/latest/ug/transcribe_example_transcribe_StartTranscriptionJob_section.html
# 
# For our sample we will start from an interview transcription in the file "interview.txt".

# In[24]:


f = open("interview.txt", "r").read()
print(f)

# In[25]:


prompt_data =f"""
Human:
Generate a summary of the transcription in the <transcription></transcription> XML tags below.
Then, classify the mood of the participants according to the closest category within 'fully satisfied', 'satisfied', 'unsatisfied', 'fully unsatisfied', or 'neutral'.

<transcription>
{f}
</transcription>

Assistant:
"""

# In[26]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[27]:


response, latency = call_bedrock('anthropic.claude-instant-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------

# ## 5. Create splash pages describing upcoming promotions
# 
# **Use Case:** A company wants to create HTML pages quickly and easily for their upcoming promotions.
# 
# **Task:** Code Generation
# 
# **Prompt Technique:** Zero-shot

# In[33]:


prompt_data ="""
There is an upcoming promotion presented by the Star Insurance company.
The promotion is targeting young audience in the age range between 18 and 40 who drives a car.
The promotion consists of a 20% discount when contracting the auto insurance policy.
There will be additional discounts if the customer is coming from another insurance company.
The promotion is part of the Summer Discounts of the company.
The promotion is available from June 28th to August 31st.

Based the this information, generate the HTML code for an attractive splash page for this promotion.
Include catchy phrases and an attractive image of a car.
Invite customers to contract the policy through a button in the portal.
Have the splash page use red fonts and black background, according to the company's branding.
"""

# In[34]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
#print(response, "\n\n", "Inference time:", latency)
from IPython.display import display, HTML
display(HTML(response))

# In[35]:


response, latency = call_bedrock('anthropic.claude-instant-v1', prompt_data)
#print(response, "\n\n", "Inference time:", latency)
from IPython.display import display, HTML
display(HTML(response))

# ------
