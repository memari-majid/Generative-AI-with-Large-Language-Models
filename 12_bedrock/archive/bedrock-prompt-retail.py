#!/usr/bin/env python
# coding: utf-8

# # Bedrock Prompt Examples for Media & Entertainment
# 
# In this notebook, we include different example use cases for Media & Entertainment using Amazon Bedrock.
# 
# These sample use cases involve different tasks and prompt engeering techniques, as follows:
# 1. **Generate recommendations based on metadata**
#     - **Task:** Text Generation
#     - **Prompt Technique:** Zero-shot
# 2. **Estimate audience for TV shows based on historical data**
#     - **Task:** Complex Reasoning
#     - **Prompt Technique:** Chain-of-Thoughts (CoT)
# 3. **Create a question answering assistant for an entertainment company**
#     - **Task:** Question Answering with Dialogue Asistant (without memory)
#     - **Prompt Technique:** Few-shot
# 4. **Summarize and classify content from media files transcription**
#     - **Task:** Text Summarization & Text Classification
#     - **Prompt Technique:** Zero-shot
# 5. **Create splash pages describing upcoming events**
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

# In[2]:


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

# In[3]:


bedrock.list_foundation_models()

# We will define an utility function for calling Bedrock.
# 
# This will help passing the proper body depending on the model invoked, and will store the results in a CSV file as well.

# In[3]:


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

    #Add interaction to the local CSV file...
    #column_name = ["timestamp", "modelId", "prompt", "response", "latency"] #The name of the columns
    #data = [datetime.now(), modelId, prompt_data, response, latency] #the data
    #with open('./prompt-data/prompt-data.csv', 'a') as f:
    #    writer = csv.writer(f)
    #    #writer.writerow(column_name)
    #    writer.writerow(data)
    
    return response, latency

# Now we are ready for running our examples with different models.
# 
# -----

# ## 1. Generate recommendations based on metadata
# 
# **Use Case:** A media & entertainment company wants to generate recommendations of TV shows for their audience based on some metadata of viewers, e.g. country, age-range, and theme.
# 
# **Task:** Text Generation
# 
# **Prompt Technique:** Zero-shot

# In[6]:


prompt_data ="""
Human:
Generate a list of 10 recommended Burger King meals for a customer considering the information in the <metadata></metadata> XML tags,
and include a very brief description of each recommendation.
Answer in Spanish.

<metadata>
Country is Spain
Age range between 20-30
Alergic to nuts
</metadata>

Assistant:
"""

# In[7]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[8]:


response, latency = call_bedrock('anthropic.claude-instant-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------

# ## 2. Estimate customers for food restaurants based on historical data
# 
# **Use Case:** A media & entertainment company wants to estimate the audience levels they could have for the next day based on historical information and shows metadata.
# 
# **Task:** Complex Reasoning
# 
# **Prompt Technique:** Chain-of-Thoughts (CoT)

# In[12]:


prompt_data ="""
Human: La semana pasada, los clientes de 3 cadenas de comida rápida fueron con el los siguientes datos:
- Lunes: Restaurante1 6500, Restaurante2 3200, Restaurante3 4150
- Martes: Restaurante1 6400, Restaurante2 3300, Restaurante3 4100
- Miércoles: Restaurante1 6300, Restaurante2 3400, Restaurante3 4250

Pregunta: ¿Cuántos clientes podemos esperar el próximo Viernes en la cadena Restaurante1?
Respuesta: Según los números dados y sin disponer de más información, hay una disminución diaria de 100 clientes para Restaurante1.
Si asumimos que esta tendencia continuará durante los siguientes días, podemos esperar 6200 clientes para el siguiente día que es Jueves y,
por lo tanto, 6100 clientes para el Viernes.

Pregunta: ¿Cuántos clientes podemos esperar el Sábado en cada una de las cadenas? Piensa paso a paso y da recomendaciones para incrementar los clientes
Assistant:
Respuesta:
"""

# In[13]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[14]:


response, latency = call_bedrock('anthropic.claude-instant-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------

# ## 3. Create a question answering assistant for an entertainment company
# 
# **Use Case:** An entertainment company wants to create a bot capable of answering questions about the shows available, based on the information for these shows.
# 
# **Task:** Question Answering with Dialogue Asistant (no memory)
# 
# **Prompt Technique:** Few-shot

# ### Titan

# In[115]:


prompt_data ="""
Context: The shows available are as follows
1. Circus, showing at the Plaza venue, assigned seating, live at 8pm on weekends
2. Concert, showing at the Main Teather, assigned seating, live at 10pm everyday
3. Basketball tricks, showing at the Sports venue, standing seating, live at 5pm on weekdays

Instruction: Answer any questions about the shows available. Do not make up any data. If you don't know the answer just say 'Apologies, but I don't have the answer for that. Please contact our team by phone.'

Agent: Welcome to Entertainment Tonight, how can I help you?
User: Hi, I would like to know what are the shows available please.
Agent: Of course, right now we have the Circus, the Concert, and the Basketball tricks shows.
User: Thank you. I would like to know when and where are those available please.
Agent:
"""

# In[116]:


response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ### Claude

# In[12]:


prompt_data ="""
Context: The shows available are as follows
1. Circus, showing at the Plaza venue, assigned seating, live at 8pm on weekends
2. Concert, showing at the Main Teather, assigned seating, live at 10pm everyday
3. Basketball tricks, showing at the Sports venue, standing seating, live at 5pm on weekdays

Instruction: Act as a friendly customer service agent, and answer any questions from the user about the shows available.
If you don't know the answer just say 'Sorry but I don't have the answer for that'.
After answering, always ask the user if you can help with anything else.

Assistant: Welcome to Entertainment Tonight, how can I help you?
Human: Hi, I would like to know what are the shows available please.
Assistant: Of course, right now we have the Circus, the Concert, and the Basketball tricks shows.
Human: Thank you. I would like to know when and where are those available please.
Assistant:
"""

# In[13]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[14]:


response, latency = call_bedrock('anthropic.claude-instant-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ### Jurassic-2

# In[120]:


prompt_data ="""
Context: The shows available are as follows
1. Circus, showing at the Plaza venue, assigned seating, live at 8pm on weekends
2. Concert, showing at the Main Teather, assigned seating, live at 10pm everyday
3. Basketball tricks, showing at the Sports venue, standing seating, live at 5pm on weekdays

Instruction: Answer any questions about these shows. If you don't know the answer just say 'Apologies, but I don't have the answer for that. Please contact our team by phone.'

Agent: Welcome to Entertainment Tonight, how can I help you?
User: Hi, I would like to know what are the shows available please.
Agent: Of course, right now we have the Circus, the Concert, and the Basketball tricks shows.
User: Thank you. I would like to know when and where are those available please.
Agent:
"""

# In[121]:


response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------

# ## 4. Generate content summary based on transcriptions from media files
# 
# **Use Case:** A media company needs to generate summaries of the audio transcription for audio and video files, to be sent to their content team. They also want to classify this content according to specific categories.
# 
# **Task:** Text Summarization & Text Classification
# 
# **Prompt Technique:** Zero-shot

# #### (Pre-requisite)
# 
# First, we will need to transcribe the media files. You can e.g. use Amazon Transcribe for this task following examples like this: https://docs.aws.amazon.com/code-library/latest/ug/transcribe_example_transcribe_StartTranscriptionJob_section.html
# 
# For our sample we will start from an interview transcription in the file "interview.txt".

# In[47]:


f = open("interview.txt", "r").read()
print(f)

# ### Titan

# In[142]:


prompt_data =f"""Context: {f}

User: Execute the following actions in order
1. Summarize the conversation in 3 sentences.
2. Clasify the conversation according to the closest category within drama, comedy, talk show, news, or sports.

Summary:
"""

# In[143]:


response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ### Claude

# In[129]:


prompt_data =f"""
Human:
Generate a summary of the transcription in the <transcription></transcription> XML tags below.
Then, classify the transcription according to the closest category within 'drama', 'comedy', 'talk show', 'news', or 'sports'.

<transcription>
{f}
</transcription>

Assistant:
"""

# In[130]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# In[131]:


response, latency = call_bedrock('anthropic.claude-instant-v1', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ### Jurassic-2

# In[116]:


prompt_data =f"""Context: {f}

Instructions: Execute the following actions in order
1. Summarize the conversation in a few sentences.
2. Clasify the conversation according to the closest category within drama, comedy, talk show, news, or sports.

Agent:
"""

# In[117]:


response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------

# ## 5. Create splash pages describing upcoming events
# 
# **Use Case:** A media and entertainment company wants to create HTML pages quickly and easily, to promote their upcoming events.
# 
# **Task:** Code Generation
# 
# **Prompt Technique:** Zero-shot

# ### Titan

# In[5]:


prompt_data ="""
There is an upcoming music concert presented by the company Music Promotions.
The event is targeting young audience in the age range between 18 and 40.
The event will be done in the Royal Music Teather.
There will be seat assignment and tickets can be bought trought the Music Promotions website.
The event is a concert of the band called Super Rockers.
The event is on June 30th 2023 and doors will be open since 20:00.

Based the this information, generate the HTML code for an attractive splash page promoting this event.
"""

# In[8]:


response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data)
#print(response, "\n\n", "Inference time:", latency)
from IPython.display import display, HTML
display(HTML(response))

# ### Claude

# In[128]:


prompt_data ="""
There is an upcoming music concert presented by the company Music Promotions.
The event is targeting young audience in the age range between 18 and 40.
The event will be done in the Royal Music Teather.
There will be seat assignment and tickets can be bought trought the Music Promotions website.
The event is a concert of the band called Super Rockers.
The event is on June 30th 2023 and doors will be open since 20:00.

Based the this information, generate the HTML code for an attractive splash page promoting this event.
"""

# In[9]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data)
#print(response, "\n\n", "Inference time:", latency)
from IPython.display import display, HTML
display(HTML(response))

# In[10]:


response, latency = call_bedrock('anthropic.claude-instant-v1', prompt_data)
#print(response, "\n\n", "Inference time:", latency)
from IPython.display import display, HTML
display(HTML(response))

# ### Jurassic-2

# In[11]:


prompt_data ="""
There is an upcoming music concert presented by the company Music Promotions.
The event is targeting young audience in the age range between 18 and 40.
The event will be done in the Royal Music Teather.
There will be seat assignment and tickets can be bought trought the Music Promotions website.
The event is a concert of the band called Super Rockers.
The event is on June 30th 2023 and doors will be open since 20:00.

Based the this information, generate the HTML code for an attractive splash page promoting this event.
"""

# In[12]:


response, latency = call_bedrock('ai21.j2-grande-instruct', prompt_data)
#print(response, "\n\n", "Inference time:", latency)
from IPython.display import display, HTML
display(HTML(response))

# ------

# In[ ]:


from datetime import datetime

body = json.dumps({
    "prompt": "Human: Hi \n Assistant:",
    "max_tokens_to_sample": 4096,
    "stop_sequences":["Human:"],
    "temperature":0,
    "top_p":0.9
})

for i in range(100):
    response = bedrock.invoke_model(
    body=body,
    modelId='anthropic.claude-v1'
    )
    response_body = json.loads(response.get('body').read())
    response = response_body.get('completion')
    print(i, datetime.now(), response)


# In[ ]:



