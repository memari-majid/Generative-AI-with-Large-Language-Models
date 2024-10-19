#!/usr/bin/env python
# coding: utf-8

# # Bedrock - Prompting for Finance and Insurance domain
# 
# In this notebook, we include different example use cases for Finance and Insurance domain using Amazon Bedrock.
# 
# All the input prompts and the output from the models are logged into an output csv file
# 
# First we install a few libraries required in the notebook

# In[4]:


#!unzip ../bedrock-preview-documentation/SDK/bedrock-python-sdk.zip -d /root/bedrock

#!pip install --upgrade pip
#!pip install scikit-learn seaborn

#!pwd
#!python3 -m pip install /root/bedrock/boto3-1.26.142-py3-none-any.whl
#!python3 -m pip install /root/bedrock/botocore-1.29.142-py3-none-any.whl

# Create the output prompt-data folder

# In[9]:


!mkdir -p prompt-data

# Initializations

# In[10]:


import boto3
import json
import csv
from datetime import datetime

bedrock = boto3.client(
 service_name='bedrock',
 region_name='us-east-1',
 endpoint_url='https://bedrock.us-east-1.amazonaws.com'
)

output_csv_file = "./prompt-data/prompt-data-finance.csv"

# List available Bedrock models

# In[7]:


bedrock.list_foundation_models()

# Generic function to invoke Bedrock models and store the output

# In[7]:


def call_bedrock(modelId, prompt_data, out_file):
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
    data = [datetime.now(), modelId, prompt_data, response, latency] #the data
    with open(out_file, 'a') as f:
        writer = csv.writer(f)
        #writer.writerow(column_name)
        writer.writerow(data)
    
    return response, latency

# ## General Question Answering1

# ### Titan

# In[125]:


# Works fine
prompt_data ="""
Generate a list of main drivers of a company's stock price considering the following data, and include a short description of each.

Country is US
List of recommendations:
"""

# In[126]:


response, latency = call_bedrock('amazon.titan-tg1-large', prompt_data, output_csv_file)
print(response, "\n\n", "Inference time:", latency)

# ### Claude

# In[101]:


# Works fine
prompt_data ="""User:
Generate a list of key drivers of a company's stock price considering the following data, and include a short description of each.
Country is US
Assistant:
"""

# In[102]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data, output_csv_file)
print(response, "\n\n", "Inference time:", latency)

# ### Jurassic-2

# In[127]:


# Works fine
prompt_data ="""
Generate a list of main drivers of a company's stock price considering the following data, and include a short description of each.

Country is US
"maxTokens": 2500
"""

# In[128]:


response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt_data, output_csv_file)
print(response, "\n\n", "Inference time:", latency)

# ## General Question Answering 2

# In[79]:


# Works fine
prompt_data ="""User:
Generate a list of main types of insurance considering the data in the <metadata></metadata> XML tags, and include a short description of each.
<metadata>
Country is US
List should include auto insurance
</metadata>
Assistant:
"""

# In[80]:


# Works fine
prompt_data ="""User:
Generate a list of main types of insurance considering the following data, and include a short description of each.
Country is US
List should include auto insurance
Assistant:
"""

# In[81]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data, output_csv_file)
print(response, "\n\n", "Inference time:", latency)

# ## General Question Answering 3

# In[95]:


# Works fine
prompt_data ="""User:
Generate a list of main reasons why is insurance important considering the following data, and include a short description of the benefits of each.
Country is US
Assistant:
"""

# In[96]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data, output_csv_file)
print(response, "\n\n", "Inference time:", latency)

# ## Question Answering with context

# In[157]:


# Works fine
prompt_data ="""User:
Answer all questions one by one based on the following context:

XYZ, INC. reported consolidated net earnings of $8,299,000 in 2022 compared to $9,193,000 in 2021. This represents $3.92 earnings per share in 2022 compared to $4.24 earnings per share in 2021.
XYZ Board of Directors continued to repurchase shares through March 16, 2022 under the previously authorized stock repurchase initiative. 45,592 shares were repurchased in 2022 at an average purchase price of $42.97. XYZ repurchased approximately 2.1% of outstanding shares, providing greater return on equity to owners of shares. XYZ ended the fiscal year with 2,104,717 shares outstanding.
On October 1, 2022, XYZ holding company of DEF Bank, successfully completed a transaction whereby XYZ acquired DEF Bank. DEF Bank provides a comprehensive suite of business, personal and retail banking products. At acquisition, DEF Bank had $239 million in assets, $203 million in net loans and $205 million in deposits. Combined assets at acquisition were $1.3 billion.
On December 15, 2022, XYZ rewarded shareholders with a cash dividend of $0.68 per share. XYZ has consistently provided shareholders an annual dividend since 1989.
Book value on December 31, 2022, was $44.41 compared to $49.51 on December 31, 2021. The Federal Reserve’s Federal Open Market Committee (FOMC) increased rates at the fastest pace since 1994, increasing rates seven times during the year; including four successive rate increases of 75 basis points. This rising interest rate environment led to unrealized loss positions in TR Bank’s bond portfolio, negatively impacting book value. Because TR Bank maintains a strong liquidity position and contingency funding plans, TR Bank has no plans to sell securities in an unrealized loss position.
XYZ’s shares closed the year at $41.51, a decrease of 2.67%. Bank stocks traded in line with the S&P 500 throughout 2022, with the S&P 500 bank index down approximately 24%. The S&P 500 was down 19%, its biggest annual percentage drop since 2008. 

Assistant:
What amount of assets had DEF Bank?
What was the closing price of XYZ's shares at the end of the year? 
"""

# In[158]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data, output_csv_file)
print(response, "\n\n", "Inference time:", latency)

# ## Summarisation

# In[230]:


# Works fine
prompt_data ="""User:
Summarise the following content in three sentences.

XYZ, INC. reported consolidated net earnings of $8,299,000 in 2022 compared to $9,193,000 in 2021. This represents $3.92 earnings per share in 2022 compared to $4.24 earnings per share in 2021.
XYZ Board of Directors continued to repurchase shares through March 16, 2022 under the previously authorized stock repurchase initiative. 45,592 shares were repurchased in 2022 at an average purchase price of $42.97. XYZ repurchased approximately 2.1% of outstanding shares, providing greater return on equity to owners of shares. XYZ ended the fiscal year with 2,104,717 shares outstanding.
On October 1, 2022, XYZ holding company of DEF Bank, successfully completed a transaction whereby XYZ acquired DEF Bank. DEF Bank provides a comprehensive suite of business, personal and retail banking products. At acquisition, DEF Bank had $239 million in assets, $203 million in net loans and $205 million in deposits. Combined assets at acquisition were $1.3 billion.
On December 15, 2022, XYZ rewarded shareholders with a cash dividend of $0.68 per share. XYZ has consistently provided shareholders an annual dividend since 1989.
Book value on December 31, 2022, was $44.41 compared to $49.51 on December 31, 2021. The Federal Reserve’s Federal Open Market Committee (FOMC) increased rates at the fastest pace since 1994, increasing rates seven times during the year; including four successive rate increases of 75 basis points. This rising interest rate environment led to unrealized loss positions in TR Bank’s bond portfolio, negatively impacting book value. Because TR Bank maintains a strong liquidity position and contingency funding plans, TR Bank has no plans to sell securities in an unrealized loss position.
XYZ’s shares closed the year at $41.51, a decrease of 2.67%. Bank stocks traded in line with the S&P 500 throughout 2022, with the S&P 500 bank index down approximately 24%. The S&P 500 was down 19%, its biggest annual percentage drop since 2008. 

Assistant:
"""

# In[231]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data, output_csv_file)
print(response, "\n\n", "Inference time:", latency)

# In[227]:


response, latency = call_bedrock('ai21.j2-jumbo-instruct', prompt_data, output_csv_file)
print(response, "\n\n", "Inference time:", latency)

# -----

# In[4]:


prompt_data="""
Context: Actua como un agente de servicio al cliente de un banco. Responde en catalán.
Human: Considera la conversación telefónica en los tags XML <conversacion></conversacion> y recomienda la próxima respuesta que debe dar el agente de servicio al cliente.

<conversacion>
Agent: Buenos días, gracias por llamar a CaixaBank, ¿cómo puedo ayudarle?
Customer: Hola, llamo porque mi tarjeta de crédito ha sido bloqueada
Agent: Primero necesito verificar unos datos de identidad, por favor indique su código
Customer: Mi código es ********
Agent: Gracias. Ahora por favor indíqueme por qué piensa que su tarjeta ha sido bloqueada
Customer: He intentado hacer un pago y no lo ha aceptado
</conversacion>

Assistant:
"""

# In[11]:


response, latency = call_bedrock('anthropic.claude-v1', prompt_data, output_csv_file)
print(response, "\n\n", "Inference time:", latency)

# In[ ]:



