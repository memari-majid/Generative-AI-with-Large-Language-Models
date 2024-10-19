#!/usr/bin/env python
# coding: utf-8

# # Chain of Thought Prompt Engineering
# 
# References: 
# - https://medium.com/nlplanet/two-minutes-nlp-making-large-language-models-reason-with-chain-of-thought-prompting-401fd3c964d0
# - https://arxiv.org/pdf/2201.11903.pdf

# In[ ]:


# !pip install sagemaker --quiet --upgrade --force-reinstall --quiet
# !pip install ipywidgets==7.0.0 --quiet

# In[3]:


model_id, model_version = "huggingface-llm-falcon-40b-instruct-bf16", "*"

# In[4]:


%%time
from sagemaker.jumpstart.model import JumpStartModel

my_model = JumpStartModel(model_id=model_id)
predictor = my_model.deploy()

# ### 1.1. Changing instance type
# ---
# 
# 
# Models have been tested on the following instance types:
# 
#  - Falcon 7B and 7B instruct: `ml.g5.2xlarge`, `ml.g5.2xlarge`, `ml.g5.4xlarge`, `ml.g5.8xlarge`, `ml.g5.16xlarge`, `ml.g5.12xlarge`, `ml.g5.24xlarge`, `ml.g5.48xlarge`, `ml.p4d.24xlarge`
#  - Falcon 40B and 40B instruct: `ml.g5.12xlarge`, `ml.g5.48xlarge`, `ml.p4d.24xlarge`
# 
# If an instance type is not available in you region, please try a different instance. You can do so by specifying instance type in the JumpStartModel class.
# 
# `my_model = JumpStartModel(model_id="huggingface-llm-falcon-40b-instruct-bf16", instance_type="ml.g5.12xlarge")`
# 
# ---

# ### 1.2. Changing number of GPUs
# ---
# Falcon models are served with HuggingFace (HF) LLM DLC which requires specifying number of GPUs during model deployment. 
# 
# **Falcon 7B and 7B instruct:** HF LLM DLC currently does not support sharding for 7B model. Thus, even if more than one GPU is available on the instance, please do not increase number of GPUs. 
# 
# **Falcon 40B and 40B instruct:** By default number of GPUs are set to 4. However, if you are using `ml.g5.48xlarge` or `ml.p4d.24xlarge`, you can increase number of GPUs to be 8 as follows: 
# 
# `my_model = JumpStartModel(model_id="huggingface-llm-falcon-40b-instruct-bf16", instance_type="ml.g5.48xlarge")`
# 
# `my_model.env['SM_NUM_GPUS'] = '8'`
# 
# `predictor = my_model.deploy()`
# 
# 
# ---

# In[5]:


endpoint_name = predictor.endpoint_name

# In[6]:


import sagemaker
import boto3
sess = sagemaker.Session()
import json

sm_client = boto3.client("sagemaker")
smr_client = boto3.client("sagemaker-runtime")

# In[7]:


parameters = {
    "early_stopping": True,
    "length_penalty": 2.0,
    "max_new_tokens": 50,
    "temperature": .1,
    "min_length": 10,
    "no_repeat_ngram_size": 3,
}

# ## Mathematical Reasoning 
# ### Zero-shot Prompting

# In[8]:


payload = """
QUESTION: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can have 3 tennis balls. How many tennis balls does he have now?
ANSWER: The answer is 11.

QUESTION: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business.
How many hours a week does he spend taking care of dogs?
ANSWER:

"""

# In[9]:


response_model = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps(
        {
            "inputs": payload,
            "parameters": parameters,
        }
    ),
    ContentType="application/json",
)

response_model["Body"].read().decode("utf8")

# ### With Chain of Thought Prompting

# In[10]:


payload = """
QUESTION: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?
ANSWER: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

QUESTION: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. 
How many hours a week does he spend taking care of dogs?
ANSWER:

"""

# In[11]:


response_model = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps(
        {
            "inputs": payload,
            "parameters": parameters,
        }
    ),
    ContentType="application/json",
)

response_model["Body"].read().decode("utf8")

# ## Advanced Mathematical Reasoning - with chain of thought prompting

# In[12]:


payload = """QUESTION: Ducks need to eat 3.5 pounds of insects each week to survive. 
If there is a flock of ten ducks, how many pounds of insects do they need per day?
ANSWER: Ducks need 3.5 pounds of insects each week. If there is a flock of 10 ducks, then they need 3.5 x 10 = 35 pounds of insects each week. If they need 35 pounds of insects each week, then they need 35 / 7 = 5 pounds of insects each day. The answer is 5. 

QUESTION: It takes Matthew 3 minutes to dig a small hole for shrubs and 10 minutes to dig a large hole for trees. 
How many hours will it take him to dig 30 small holes and 15 large holes?
ANSWER: It takes Matthew 3 minutes to dig a small hole and 10 minutes to dig a large hole. So, it takes Matthew 3 x 30 = 90 minutes to dig 30 small holes. It takes Matthew 10 x 15 = 150 minutes to dig 15 large holes. So, it takes Matthew 90 + 150 = 240 minutes to dig 30 small holes and 15 large holes. 240 minutes is 4 hours. The answer is 4 hours. 

QUESTION: I have 10 liters of orange drink that are two-thirds water and I wish to add it to 15 liters of pineapple drink that is three-fifths water. But as I pour it, I spill one liter of the orange drink. How much water is in the remaining 24 liters?
ANSWER:

"""

# In[13]:


response_model = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps(
        {
            "inputs": payload,
            "parameters": parameters,
        }
    ),
    ContentType="application/json",
)

response_model["Body"].read().decode("utf8")

# As you can see in the above example for complex mathematical reasoning the models might not give you the right predicted output. 
# The correnct answer is: 
# 
# "The orange drink is 10liters, 1 liter was dropped, remaining drink has 9 * 2/3 = 6 liters of water. The pineapple drink is 15 x 3 / 5 = 9 liter of water in it. The total water in the orange and pineapple drinks is 15"

# ## Symbolic Reasoning
# For symbolic reasoning, consider the tasks of last letter concatenation, reverse list, and coin flip shown in the next image.

# ### Zero shot prompting

# ### Last Letter Concatenation

# In[14]:


payload = """QUESTION: Take the last letters of the words in "Elon Musk" and con-catenate them.
ANSWER: 

"""

# # [{"generated_text":"musk elon n"}]


# In[15]:


response_model = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps(
        {
            "inputs": payload,
            "parameters": parameters,
        }
    ),
    ContentType="application/json",
)

response_model["Body"].read().decode("utf8")

# ### With Chain of thought prompting

# In[16]:


payload = """QUESTION: Take the last letters of the words in "Elon Musk" and con-catenate them.
ANSWER: The last letter of "Elon" is "n". The last letter of "Musk" is "k'. Concatenating them is "nk". So the answer is nk.

QUESTION: Take the last letters of the words in "Chris Fregly" and con-catenate them.
ANSWER: The last letter of "Chris" is "s". The last letter of "Fregly" is "y". Concatenating them is "sy". So the answer is sy. 

QUESTION: Take the last letters of the words in "John Doe" and con-catenate them.
ANSWER:

"""

# In[17]:


response_model = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps(
        {
            "inputs": payload,
            "parameters": parameters,
        }
    ),
    ContentType="application/json",
)

response_model["Body"].read().decode("utf8")

# ### Reverse List
# 
# ### Zero shot prompting

# In[18]:


payload = """QUESTION: Reverse the sequence "glasses, pen, alarm, license".
ANSWER: 

"""

# In[19]:


response_model = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps(
        {
            "inputs": payload,
            "parameters": parameters,
        }
    ),
    ContentType="application/json",
)

response_model["Body"].read().decode("utf8")

# ### With Chain of Thought prompting

# In[20]:


payload = """
QUESTION: Reverse the sequence "glasses, pen, alarm, license".
ANSWER: First is glasses. Second is pen. Third is alarm. Fourth is license. Now to reverse, change the order to: Fourth is license.
Third is alarm. Second is pen. First is glasses. So the answer is
"license, alarm, pen, glasses".

QUESTION: Reverse the sequence "telephone, clock, board, spectacles".
ANSWER: First is telephone. Second is clock. Third is board. Fourth is spectacles. Now to reverse, change the order to: Fourth is spectacles.
Third is board. Second is clock. First is telephone. So the answer is
"spectacles, board, clock, telephone".

QUESTION: Reverse the sequence "cup, plate, food, fruits".
ANSWER:

"""

# # [{"generated_text":"fruits, food, plate, cup\\" is correct."}]'

# In[21]:


response_model = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps(
        {
            "inputs": payload,
            "parameters": parameters,
        }
    ),
    ContentType="application/json",
)

response_model["Body"].read().decode("utf8")

# ### Coin Flip
# ### Zero shot prompting

# In[22]:


payload = """
QUESTION:  A coin is heads up. John does not flip the coin. S
halonda does not flip the coin. Is the coin still heads up?
ANSWER: 

"""

# In[23]:


response_model = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps(
        {
            "inputs": payload,
            "parameters": parameters,
        }
    ),
    ContentType="application/json",
)

response_model["Body"].read().decode("utf8")

# In[24]:


payload = """
QUESTION: A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin. Is the coin still heads up?
ANSWER: The coin was flipped by Maybelle. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer
is no.

QUESTION:  A coin is heads up. John does not flip the coin. Shalonda does not flip the coin. Is the coin still heads up?
ANSWER:

"""

# In[25]:


response_model = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps(
        {
            "inputs": payload,
            "parameters": parameters,
        }
    ),
    ContentType="application/json",
)

response_model["Body"].read().decode("utf8")

# In[ ]:



