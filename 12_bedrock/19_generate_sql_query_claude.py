#!/usr/bin/env python
# coding: utf-8

# # Invoke Bedrock model for SQL Query Generation
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*

# ## Introduction
# 
# In this notebook we show you how to use a LLM to generate SQL Query to analyze Sales data.
# 
# We will use Bedrock's Claude V2 model using the Boto3 API. 
# 
# The prompt used in this example is called a zero-shot prompt because we are not providing any examples of text other than the prompt.
# 
# **Note:** *This notebook can be run within or outside of AWS environment.*
# 
# #### Context
# To demonstrate the SQL code generation capability of Amazon Bedrock, we will explore the use of Boto3 client to communicate with Amazon Bedrock API. We will demonstrate different configurations available as well as how simple input can lead to desired outputs.
# 
# #### Pattern
# We will simply provide the Amazon Bedrock API with an input consisting of a task, an instruction and an input for the model under the hood to generate an output without providing any additional example. The purpose here is to demonstrate how the powerful LLMs easily understand the task at hand and generate compelling outputs.
# 
# ![](./images/sql-query-generation.png)
# 
# #### Use case
# Let's take the use case to generate SQL queries to analyze sales data, focusing on trends, top products and average sales.
# 
# #### Persona
# Maya is a business analyst, at AnyCompany primarily focusing on sales and inventory data. She is transitioning from Speadsheet analysis to data-driven analysis and want to use SQL to fetch specific data points effectively. She wants to use LLMs to generate SQL queries for her analysis. 
# 
# #### Implementation
# To fulfill this use case, in this notebook we will show how to generate SQL queries. We will use the Anthropic Claude v2 model using the Amazon Bedrock API with Boto3 client. 

# ## Setup

# In[2]:


%pip install --no-build-isolation --force-reinstall \
    "boto3>=1.28.57" \
    "awscli>=1.29.57" \
    "botocore>=1.31.57"

%pip install --quiet langchain==0.0.309

# In[3]:


import warnings
warnings.filterwarnings('ignore')

# In[4]:


#### Un comment the following lines to run from your local environment outside of the AWS account with Bedrock access

#import os
#os.environ['BEDROCK_ASSUME_ROLE'] = '<YOUR_VALUES>'
#os.environ['AWS_PROFILE'] = '<YOUR_VALUES>'

# In[5]:


import boto3
import json 

bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# ## Generate SQL Query
# 
# Following on the use case explained above, let's prepare an input for  the Amazon Bedrock service to generate SQL query.

# In[6]:


# create the prompt to generate SQL query
prompt_data = """

Human: AnyCompany has a database with a table named sales_data containing sales records. The table has following columns:
- date (YYYY-MM-DD)
- product_id
- price
- units_sold

Can you generate SQL queries for the below: 
- Identify the top 5 best selling products by total sales for the year 2023
- Calculate the monthly average sales for the year 2023

Assistant:
"""


# Let's start by using the Anthorpic Claude v2 model. 

# In[7]:


# Claude - Body Syntex
body = json.dumps({
                    "prompt": prompt_data,
                    "max_tokens_to_sample":4096,
                    "temperature":0.5,
                    "top_k":250,
                    "top_p":0.5,
                    "stop_sequences": ["\n\nHuman:"]
                  }) 

# #### Invoke the Bedrock's Claude Large Large language model

# First, we explore how the model generates an output based on the prompt created earlier.
# 
# ##### Complete Output Generation

# In[9]:


modelId = 'anthropic.claude-v2' # change this to use a different version from the model provider
accept = 'application/json'
contentType = 'application/json'

response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
response_body = json.loads(response.get('body').read())

print(response_body.get('completion'))

# ### Advanced Example
# #### Understanding Hospital's Patient Management System through SQL

# In[10]:


# create the prompt
prompt_sql_data = """

Human: You're provided with a database schema representing any hospital's patient management system.
The system holds records about patients, their prescriptions, doctors, and the medications prescribed.

Here's the schema:

```sql
CREATE TABLE Patients (
    PatientID int,
    FirstName varchar(50),
    LastName varchar(50),
    DateOfBirth datetime,
    Gender varchar(10),
    PRIMARY KEY (PatientID)
);

CREATE TABLE Doctors (
    DoctorID int,
    FirstName varchar(50),
    LastName varchar(50),
    Specialization varchar(50),
    PRIMARY KEY (DoctorID)
);

CREATE TABLE Prescriptions (
    PrescriptionID int,
    PatientID int,
    DoctorID int,
    DateIssued datetime,
    PRIMARY KEY (PrescriptionID)
);

CREATE TABLE Medications (
    MedicationID int,
    MedicationName varchar(50),
    Dosage varchar(50),
    PRIMARY KEY (MedicationID)
);

CREATE TABLE PrescriptionDetails (
    PrescriptionDetailID int,
    PrescriptionID int,
    MedicationID int,
    Quantity int,
    PRIMARY KEY (PrescriptionDetailID)
);
```

Write a SQL query that fetches all the patients who were prescribed more than 5 different medications on 2023-04-01.

Assistant:
"""


# In[11]:


# Claude - Body Syntex
body = json.dumps({
                    "prompt": prompt_sql_data,
                    "max_tokens_to_sample":4096,
                    "temperature":0.5,
                    "top_k":250,
                    "top_p":0.5,
                    "stop_sequences": ["\n\nHuman:"]
                  }) 

# In[12]:


modelId = 'anthropic.claude-v2' # change this to use a different version from the model provider
accept = 'application/json'
contentType = 'application/json'

response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
response_body = json.loads(response.get('body').read())

print(response_body.get('completion'))

# In[ ]:



