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

# This notebook uses Amazon Bedrock, make sure you have access to the service and have the dependencies installed if required. You can check details for this in the [Amazon Bedrock Workshop](https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/00_Intro/bedrock_boto3_setup.ipynb).

# First, let's setup the Bedrock client.

# In[1]:


import sys, os

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock_utils

# ---- ⚠️ Un-comment and edit the below lines as needed for your AWS setup ⚠️ ----

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # E.g. "us-east-1"
# os.environ["AWS_PROFILE"] = "<YOUR_PROFILE>"
os.environ["BEDROCK_ASSUME_ROLE"] = "arn:aws:iam::114798159153:role/BedrockRemote"  # E.g. "arn:aws:..."
# os.environ["BEDROCK_ENDPOINT_URL"] = "<YOUR_ENDPOINT_URL>"  # E.g. "https://..."

bedrock = bedrock_utils.get_bedrock(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)

# In[2]:


bedrock.list_foundation_models()

# **Note:** We'll use a utility function 'call_bedrock' for invoking Bedrock with the proper parameters.
# 
# It accepts the following arguments:
# * Bedrock client (leave as 'bedrock' in this notebook)
# * Prompt
# * ModelId (if not specified, defaults to Claude v2)
# * Temperature (if not specified, defaults to 0)
# * Top P (if not specified, defaults to 0.9)
# 
# You can adjust any other parameters by modifying the function [here](./utils/bedrock_utils.py)

# ## 1. Generate recommendations based on metadata
# 
# **Use Case:** A media & entertainment company wants to generate recommendations of TV shows for their audience based on some metadata of viewers, e.g. country, age-range, and theme.
# 
# **Task:** Text Generation
# 
# **Prompt Technique:** Zero-shot

# In[3]:


prompt_data ="""
Human:
Generate a list of 10 recommended TV shows to watch considering the information in the <metadata></metadata> XML tags, and include a very brief description of each recommendation.

<metadata>
Country is UK
Age range between 20-30
Shows must be about sports
</metadata>

Assistant:
"""

# In[4]:


response, latency = bedrock_utils.call_bedrock(bedrock, prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------

# ## 2. Estimate audience for TV shows based on historical data
# 
# **Use Case:** A media & entertainment company wants to estimate the audience levels they could have for the next day based on historical information and shows metadata.
# 
# **Task:** Complex Reasoning
# 
# **Prompt Technique:** Chain-of-Thoughts (CoT)

# In[12]:


prompt_data ="""
Human: Last week, the viewers of 3 television networks were with the following data:
- Monday: SportsTV 6500, NewsTV 3200, EntertainmentTV 4150
- Tuesday: SportsTV 6400, NewsTV 3300, EntertainmentTV 4100
- Wednesday: SportsTV 6300, NewsTV 3400, EntertainmentTV 4250

Question: How many viewers can we expect next Friday on SportsTV?
Answer: According to the numbers given and without having more information, there is a daily decrease of 100 viewers for SportsTV.
If we assume that this trend will continue for the next few days, we can expect 6200 viewers for the next day which is Thursday, and
therefore 6100 viewers for Friday.

Question: How many viewers can we expect on Saturday in each of the channels? think step-by-step, and provide recommendations for increasing the viewers
Assistant:
Answer:
"""

# In[13]:


response, latency = bedrock_utils.call_bedrock(bedrock, prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------

# ## 3. Create a question answering assistant for an entertainment company
# 
# **Use Case:** An entertainment company wants to create a bot capable of answering questions about the shows available, based on the information for these shows.
# 
# **Task:** Question Answering with Dialogue Asistant (no memory)
# 
# **Prompt Technique:** Few-shot

# In[14]:


prompt_data ="""
Context: The shows available are as follows
1. Circus, showing at the Plaza venue, assigned seating, live at 8pm on weekends
2. Concert, showing at the Main Teather, assigned seating, live at 10pm everyday
3. Basketball tricks, showing at the Sports venue, standing seating, live at 5pm on weekdays

Instruction: Answer any questions about the shows available. If you don't know the answer just say 'Apologies, but I don't have the answer for that. Please contact our team by phone.'

Assistant: Welcome to Entertainment Tonight, how can I help you?
Human: Hi, I would like to know what are the shows available please.
Assistant: Of course, right now we have the Circus, the Concert, and the Basketball tricks shows.
Human: Thank you. I would like to know when and where are those available please.
Assistant:
"""

# In[15]:


response, latency = bedrock_utils.call_bedrock(bedrock, prompt_data)
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

# In[16]:


f = open("interview.txt", "r").read()
print(f)

# In[17]:


prompt_data =f"""
Human:
Generate a summary of the transcription in the <transcription></transcription> XML tags below.
Then, classify the transcription according to the closest category within 'drama', 'comedy', 'talk show', 'news', or 'sports'.

<transcription>
{f}
</transcription>

Assistant:
"""

# In[18]:


response, latency = bedrock_utils.call_bedrock(bedrock, prompt_data)
print(response, "\n\n", "Inference time:", latency)

# ------

# ## 5. Create splash pages describing upcoming events
# 
# **Use Case:** A media and entertainment company wants to create HTML pages quickly and easily, to promote their upcoming events.
# 
# **Task:** Code Generation
# 
# **Prompt Technique:** Zero-shot

# In[21]:


prompt_data ="""
There is an upcoming music concert presented by the company Music Promotions.
The event is targeting young audience in the age range between 18 and 40.
The event will be done in the Royal Music Teather.
There will be seat assignment and tickets can be bought trought the Music Promotions website.
The event is a concert of the band called Super Rockers.
The event is on June 30th 2023 and doors will be open since 20:00.

Based the this information, generate the HTML code for an attractive splash page promoting this event.
"""

# In[22]:


response, latency = bedrock_utils.call_bedrock(bedrock, prompt_data)
#print(response, "\n\n", "Inference time:", latency)
from IPython.display import display, HTML
display(HTML(response))

# ------
