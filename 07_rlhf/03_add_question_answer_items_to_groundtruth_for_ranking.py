#!/usr/bin/env python
# coding: utf-8

# # Add items for human feedback
# 
# ![Pipeline](img/generative_ai_pipeline_rlhf_plus.png)
# 
# ![RLHF](img/rlhf_qa.png)

# In[3]:


import psutil

notebook_memory = psutil.virtual_memory()
print(notebook_memory)

if notebook_memory.total < 32 * 1000 * 1000 * 1000:
    print('*******************************************')    
    print('YOU ARE NOT USING THE CORRECT INSTANCE TYPE')
    print('PLEASE CHANGE INSTANCE TYPE TO  m5.2xlarge ')
    print('*******************************************')
else:
    correct_instance_type=True

# In[4]:


import boto3
import sagemaker
import pandas as pd

sess = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

# In[5]:


import io
import json
import uuid
import time
import boto3
import botocore

# Amazon Python SDK clients
sagemaker = boto3.client("sagemaker", region)
#comprehend = boto3.client("comprehend", region)
a2i = boto3.client("sagemaker-a2i-runtime")
s3 = boto3.client("s3", region)

# # Retrieve the `augmented_ai_flow_definition_arn` Created Previously

# In[6]:


%store -r augmented_ai_flow_definition_arn

# In[7]:


print(augmented_ai_flow_definition_arn)

# In[8]:


items = [
    {
        "prompt": """
            Chris: Hey Antje and Shelbee! Do you want to write a book on generative AI?
            Antje: That sounds fun. What do you think, Shelbee?
            Shelbee: Of course! Should we title the book, “Generative AI on AWS”?
            Chris: Yes!
            Antje: Yes!
        """,
        "responses": [
            """
            Chris, Shelbee, and Antje agree to write a book titled, “Generative AI on AWS”.           
            """,
            """
            Chris asks Antje and Shelbee if they want to write a book on generative AI. They agree to write the book.
            """,
            """
            Chris asks Antje and Shelbee to write a book. They agree.            
            """,
            """
            Chris, Antje, and Shelbee decide not to write a book.            
            """            
        ]
    }
]

# In[9]:


human_loops_started = []

for item in items:
    print(f'Processing item: "{item}"')

    humanLoopName = str(uuid.uuid4())
    inputContent = {"taskObject": item}
    start_loop_response = a2i.start_human_loop(
        HumanLoopName=humanLoopName,
        FlowDefinitionArn=augmented_ai_flow_definition_arn,
        HumanLoopInput={"InputContent": json.dumps(inputContent)},
    )

    human_loops_started.append(humanLoopName)

    print(f"*** ==> Starting human loop with name: {humanLoopName}  \n")

# In[10]:


%store human_loops_started

# # Check Status of Human Loop

# In[11]:


completed_human_loops = []
for human_loop_name in human_loops_started:
    resp = a2i.describe_human_loop(HumanLoopName=human_loop_name)
    print(f"HumanLoop Name: {human_loop_name}")
    print(f'HumanLoop Status: {resp["HumanLoopStatus"]}')
    print(f'HumanLoop Output Destination: {resp["HumanLoopOutput"]}')
    print("")

    if resp["HumanLoopStatus"] == "Completed":
        completed_human_loops.append(resp)

# # Wait For Workers to Complete Their Human Loop Tasks
# 
# Navigate to the link below and login with your email and password that you used when you set up the Private Workforce.

# In[12]:


%store -r augmented_ai_workteam_arn

# In[13]:


print(augmented_ai_workteam_arn)

# In[14]:


workteam_name = augmented_ai_workteam_arn[augmented_ai_workteam_arn.rfind("/") + 1 :]
print(workteam_name)
print("Navigate to the private worker portal and complete the human loop.")
print("Make sure you have invited yourself to the workteam and received the signup email.")
print("Note:  Check your spam filter if you have not received the email.")
print("")
print("https://" + sagemaker.describe_workteam(WorkteamName=workteam_name)["Workteam"]["SubDomain"])

# # _YOU MUST LABEL THE DATA BY CLICKING THE LINK ABOVE BEFORE CONTINUING!!_

# # Start Labeling
# 
# <img src="img/augmented-comprehend-custom-start-working.png" width="80%" align="left">

# # Select Label
# 
# <img src="img/augmented-comprehend-custom-select-label.png" width="80%" align="left">

# # Loop is Completed
# 
# <img src="img/augmented-comprehend-custom-finished-task.png" width="80%" align="left">

# # Verify the Human Loops are Completed

# In[15]:


workteam_name = augmented_ai_workteam_arn[augmented_ai_workteam_arn.rfind("/") + 1 :]
print(workteam_name)
print("Navigate to the private worker portal and complete the human loop.")
print("Make sure you have invited yourself to the workteam and received the signup email.")
print("Note:  Check your spam filter if you have not received the email.")
print("")
print("https://" + sagemaker.describe_workteam(WorkteamName=workteam_name)["Workteam"]["SubDomain"])

# In[16]:


import time

completed_human_loops = []
for human_loop_name in human_loops_started:
    resp = a2i.describe_human_loop(HumanLoopName=human_loop_name)
    print(f"HumanLoop Name: {human_loop_name}")
    print(f'HumanLoop Status: {resp["HumanLoopStatus"]}')
    print(f'HumanLoop Output Destination: {resp["HumanLoopOutput"]}')
    print("")
    while resp["HumanLoopStatus"] != "Completed":
        print(f"Waiting for HumanLoop to complete.")
        time.sleep(10)
        resp = a2i.describe_human_loop(HumanLoopName=human_loop_name)
    if resp["HumanLoopStatus"] == "Completed":
        completed_human_loops.append(resp)
        print(f"Completed!")
        print("")

# # View Human Labels  

# Once the work is complete, Amazon GroundTruth stores the results in the specified S3 bucket and sends a Cloudwatch Event.  Here is a sample item labeled with GroundTruth in `jsonlines` format:
# ```
# {
#  "inputContent": {"taskObject": {
#                          "prompt": "Who is Angela Merkel's favorite President of the United States?",
#                          "responses": ["George Clinton", "Barack Obama"]}
#                  },
#  "humanAnswers": [{"answerContent": {
#                         "ranking_1": "1", # ranking for 1st response (1 is High)
#                         "ranking_2": "2"  # ranking for 2nd response (2 is Low)
#                  }}]
# }
# ```

# # Prepare human-labeled data for RL/PPO training
# Retrieve from GrountTruth and convert to a binary reward (-1, 1) for all rankings as follows:
# 
# From this:
# ```
# prompt                                                              response           ranking
# 
# Who is Angela Merkel's favorite President of the United States?     George Clinton     1   # High
# Who is Angela Merkel's favorite President of the United States?     Barack Obama       2   # Low
# ```
# 
# To this:
# ```
# prompt                                                              response           reward
# 
# Who is Angela Merkel's favorite President of the United States?     George Clinton     0   # Low reward
# Who is Angela Merkel's favorite President of the United States?     Barack Obama       1   # High reward
# ```
# 
# To this:
# ```
# prompt                                                              response                               highest_ranked_response
# 
# Who is Angela Merkel's favorite President of the United States?     ["George Clinton", "Barack Obama"]     [0, 1]
# ```

# # _Note:  If nothing is showing up below, you need to return to finish the previous notebook by labeling the data in Ground Truth!!_

# In[17]:


import re
from pprint import pprint

human_feedback_items = []

for resp in completed_human_loops:
    human_feedback_s3_uri = resp["HumanLoopOutput"]["OutputS3Uri"]
    split_string = re.split("s3://" + bucket + "/", resp["HumanLoopOutput"]["OutputS3Uri"])
    key = split_string[1]
    
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    json_output = json.loads(content)

    prompt = json_output["inputContent"]['taskObject']['prompt']
    responses = json_output["inputContent"]['taskObject']['responses']
    response_1_ranking = json_output["humanAnswers"][0]["answerContent"]['response_1_ranking']
    response_2_ranking = json_output["humanAnswers"][0]["answerContent"]['response_2_ranking']
    
    human_feedback_item_1 = (prompt, responses[0], response_1_ranking)
    human_feedback_items.append(human_feedback_item_1)
    human_feedback_item_2 = (prompt, responses[1], response_2_ranking)
    human_feedback_items.append(human_feedback_item_2)

# In[18]:


df_human_feedback_items = pd.DataFrame(human_feedback_items, columns=['prompt', 'response', 'ranking'])
df_human_feedback_items.head(10)

# # Convert ranking into 0 or 1 reward

# In[19]:


num_rankings = 2
df_human_feedback_items['response'] = df_human_feedback_items['response'].apply(lambda response: str(response))
df_human_feedback_items['ranking'] = df_human_feedback_items['ranking'].apply(lambda ranking: str(abs(int(ranking) - num_rankings)))
df_human_feedback_items.head(10)

# In[20]:


df_human_feedback_items_grouped_by_prompt = df_human_feedback_items.groupby('prompt', as_index=False).agg({'prompt' : 'first', 'response' : ','.join, 'ranking' : ','.join})
df_human_feedback_items_grouped_by_prompt

# In[21]:


df_human_feedback_items_grouped_by_prompt['response'] = df_human_feedback_items_grouped_by_prompt['response'].apply(lambda response: [s for s in response.split(',')])
df_human_feedback_items_grouped_by_prompt['ranking'] = df_human_feedback_items_grouped_by_prompt['ranking'].apply(lambda ranking: [int(s) for s in ranking.split(',')])
df_human_feedback_items_grouped_by_prompt

# In[22]:


from datasets import Dataset

# Create Dataset objects (Arrow PyTables) from Pandas dataframes
human_feedback_dataset = Dataset.from_pandas(df_human_feedback_items_grouped_by_prompt)
human_feedback_dataset

# In[23]:


%store human_feedback_dataset

# # _YOU MUST LABEL THE DATA BY CLICKING THE LINK ABOVE BEFORE CONTINUING!!_

# In[24]:


%%html

<p><b>Shutting down your kernel for this notebook to release resources.</b></p>
<button class="sm-command-button" data-commandlinker-command="kernelmenu:shutdown" style="display:none;">Shutdown Kernel</button>

<script>
try {
    els = document.getElementsByClassName("sm-command-button");
    els[0].click();
}
catch(err) {
    // NoOp
}
</script>

# In[ ]:



