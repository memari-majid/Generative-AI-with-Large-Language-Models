#!/usr/bin/env python
# coding: utf-8

# To incorporate human feedback into your human review workflows, you need the following resources:
# 
# * **Workforce** to label your dataset. You can choose either the Amazon Mechanical Turk workforce, a vendor-managed workforce, or you can create your own private workforce for human reviews. Whichever workforce type you choose, Amazon Ground Truth takes care of sending tasks to workers.
# 
# * **Worker Task Template** to create a Human Task UI for the worker. The worker UI displays your input data, such as documents or images, and instructions to workers. It also provides interactive tools that the worker uses to complete your tasks. For more information, see https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-instructions-overview.html
# 
# * **Flow Definition** to create a Human Review Workflow. You use the flow definition to configure your human workforce and provide information about how to accomplish the human review task. You can create a flow definition in the SageMaker Ground Truth console or with APIs. To learn more about both of these options, see https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-create-flow-definition.html
# 
# * **Human Loop** starts your human review workflow. When you use one of the built-in task types, the corresponding AWS service creates and starts a human loop on your behalf when the conditions specified in your flow definition are met or for each object if no conditions were specified. When a human loop is triggered, human review tasks are sent to the workers as specified in the flow definition.
# 
# When using a custom task type, as this notebook will show, you start a human loop using the AWS API. When you call StartHumanLoop in your custom application, a task is sent to human reviewers.

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
a2i = boto3.client("sagemaker-a2i-runtime")
s3 = boto3.client("s3", region)

# # Setup the S3 Output Location for the Workflow Results

# In[6]:


output_path = f"s3://{bucket}/ground-truth-star-rating-results"
print(output_path)

# # Setup the Workforce and Workteam
# A workforce is the group of workers that you have selected to label your dataset. When you use a private workforce, you also create work teams, a group of workers from your workforce that are assigned to Amazon Augmented AI human review tasks. You can have multiple work teams and can assign one or more work teams to each job.
# 
# To create a new Workforce and Workteam, navigate here:
#  

# In[7]:


print(
    "https://{}.console.aws.amazon.com/sagemaker/groundtruth?region={}#/labeling-workforces/create".format(
        region, region
    )
)

# <img src="img/augmented-create-workforce.png" width="80%" align="left">

# # Look for the Email - Check Your Junk Mail Folder

# <img src="img/augmented-create-workforce-confirmation-email.png" width="60%" align="left">

# # Review the Workforce Status

# <img src="img/augmented-create-workforce-confirmation.png" width="80%" align="left">

# # Set the `workteam_arn`

# In[8]:


import boto3

account_id = boto3.client("sts").get_caller_identity().get("Account")

augmented_ai_workteam_arn = "arn:aws:sagemaker:{}:{}:workteam/private-crowd/dsoaws".format(region, account_id)

print(augmented_ai_workteam_arn)

# Visit: https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-permissions-security.html to add the necessary permissions to your role

# # Create the Human Task UI using a Worker Task Template
# 
# Create a human task UI resource, giving a UI template.  This template will be rendered to the human workers whenever human interaction is required.
# 
# Below we've provided a simple demo template that is compatible with AWS Comprehend's Detect Sentiment API.
# 
# For other pre-built UIs (70+), check: https://github.com/aws-samples/amazon-a2i-sample-task-uis

# # Ask the human to rank the star_ratings generated from the LLM (deployed in the previous step) for a given prompt
# 
# We will ask the human worker to rank the star_ratings for a given prompt based on the set of star_ratings predicted by the LLM in a previous step.  These rankings are used to train the reward model in a future step.

# In[9]:


template = r"""
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<crowd-form>
  <short-instructions>
      Select the correct ranking from the dropdown (High, Low).
  </short-instructions>

  <full-instructions header="Ranking Instructions">
      Select the correct ranking from the dropdown (High, Low).
  </full-instructions>

  <p>
      For the following question <br/><b>{{ task.input.taskObject.prompt }}</b><br/> select the ranking (Low, High) for the following answer <br/><b>{{ task.input.taskObject.responses[0] }}</b></br>
  </p>
  <div>
      <select name="response_1_ranking" required>
          <option disabled selected value> -- select an option -- </option>
          <option value="1">High</option>
          <option value="2">Low</option>
      </select>
  </div>
  <p>
      For the following question <br/><b>{{ task.input.taskObject.prompt }}</b><br/> select the ranking (Low, High) for the following answer <br/><b>{{ task.input.taskObject.responses[1] }}</b></br>
  </p>
  <div>
      <select name="response_2_ranking" required>
          <option disabled selected value> -- select an option -- </option>
          <option value="1">High</option>
          <option value="2">Low</option>
      </select>
  </div>
</crowd-form>
"""

# In[10]:


# Task UI name - this value is unique per account and region. You can also provide your own value here.
task_ui_name = "ui-ranking-" + str(uuid.uuid4())

# Create a Human Task UI resource.
human_task_ui_response = sagemaker.create_human_task_ui(HumanTaskUiName=task_ui_name, UiTemplate={"Content": template})
human_task_ui_arn = human_task_ui_response["HumanTaskUiArn"]
print(human_task_ui_arn)

# # Create a Flow Definition

# In this section, we're going to create a flow definition. Flow Definitions allow us to specify:
# 
# * The workforce that your tasks will be sent to.
# * The instructions that your workforce will receive. This is called a worker task template.
# * The configuration of your worker tasks, including the number of workers that receive a task and time limits to complete tasks.
# * Where your output data will be stored.
# 
# This demo is going to use the API, but you can optionally create this workflow definition in the console as well. 
# 
# For more details and instructions, see: https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-create-flow-definition.html.

# In[11]:


import uuid

# Flow definition name - this value is unique per account and region. You can also provide your own value here.
flow_definition_name = "fd-ranking-" + str(uuid.uuid4())

create_workflow_definition_response = sagemaker.create_flow_definition(
    FlowDefinitionName=flow_definition_name,
    RoleArn=role,
    HumanLoopConfig={
        "WorkteamArn": augmented_ai_workteam_arn,
        "HumanTaskUiArn": human_task_ui_arn,
        "TaskCount": 1,
        "TaskDescription": "Rank the answer for the given question from Low (Worst) to High (Best)",
        "TaskTitle": "Rank the answer for the given question from Low (Worst) to High (Best)",
        
    },
    OutputConfig={"S3OutputPath": output_path},
)

augmented_ai_flow_definition_arn = create_workflow_definition_response["FlowDefinitionArn"]

# # _If you see an error ^^^^ above ^^^^, you need to create your private workforce first. See the steps above. Then, re-run this cell._

# In[12]:


# Describe flow definition - status should turn to "active"
for x in range(60):
    describeFlowDefinitionResponse = sagemaker.describe_flow_definition(FlowDefinitionName=flow_definition_name)
    print(describeFlowDefinitionResponse["FlowDefinitionStatus"])
    if describeFlowDefinitionResponse["FlowDefinitionStatus"] == "Active":
        print("Flow Definition is active")
        break
    time.sleep(2)

# In[13]:


%store augmented_ai_flow_definition_arn

# In[14]:


%store augmented_ai_workteam_arn

# # Releasing Resources

# In[15]:


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
