#!/usr/bin/env python
# coding: utf-8

# # Bedrock with LangChain - Code Translation from one programming language to another
# 
# > *If you see errors, you may need to be allow-listed for the Bedrock models used by this notebook*
# 
# > *This notebook should work well with the **`Data Science 3.0`** kernel in SageMaker Studio*

# ## Introduction
# 
# In this notebook, you will learn how to translate code from one programming language to another using LLMs on Amazon Bedrock. We will demonstrate the use of LLMs as well as how to utilize LangChain framework to integrate with Bedrock.
# 
# We will use Claude v2 model of Amazon Bedrock in this lab.
# 
# **Note:** *This notebook can be run within or outside of AWS environment.*
# 
# #### Context
# In the previous example `02_code_interpret_w_langchain.ipynb`, we explored how to use LangChain framework to communicate with Amazon Bedrock API. Similar to previous example of code interpret/explain, we will use LangChain and Amazon Bedrock APIs to translate code from one legacy programming language to another.
# 
# 
# #### Pattern
# We will simply provide the LangChain implementation of Amazon Bedrock API with an input consisting of a task, an instruction and an input for the model under the hood to generate an output without providing any additional example. The purpose here is to demonstrate how the powerful LLMs easily understand the task at hand and generate compelling outputs.
# 
# ![](./images/code-translation-langchain.png)
# 
# #### Use case
# To demonstrate how you can use Amazon Bedrock LLMs to translate code from one programming language to another.
# 
# #### Persona
# Guides you through translating C++ code to Java using Amazon Bedrock and LangChain APIs. It shows techniques for prompting the model to port C++ code over to Java, handling differences in syntax, language constructs, and conventions between the languages.
# 
# #### Implementation
# To fulfill this use case, we will show you how to translate a given legacy C++ code to port to Java.  We will use the Amazon Bedrock and LangChain integration. 
# 

# ## Setup
# 
# In this notebook, we'll also install the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library which we'll use for counting the number of tokens in an input prompt.

# In[2]:


%pip install --no-build-isolation --force-reinstall \
    "boto3>=1.28.57" \
    "awscli>=1.29.57" \
    "botocore>=1.31.57"

%pip install --quiet langchain==0.0.309 "transformers>=4.24,<5"

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

# ## Invoke the Bedrock LLM Model
# 
# We'll begin with creating an instance of Bedrock class from llms. This expects a `model_id` which is the ARN of the model available in Amazon Bedrock. 
# 
# Optionally you can pass on a previously created boto3 client as well as some `model_kwargs` which can hold parameters such as `temperature`, `topP`, `maxTokenCount` or `stopSequences` (more on parameters can be explored in Amazon Bedrock console).
# 
# Available text generation models under Amazon Bedrock have the following IDs:
# 
# - amazon.titan-tg1-large
# - ai21.j2-grande-instruct
# - ai21.j2-jumbo-instruct
# - anthropic.claude-instant-v1
# - anthropic.claude-v2
# 
# Note that different models support different `model_kwargs`.

# In[6]:


from langchain.llms.bedrock import Bedrock

inference_modifier = {'max_tokens_to_sample':4096, 
                      "temperature":0.5,
                      "top_k":250,
                      "top_p":1,
                      "stop_sequences": ["\n\nHuman"]
                     }

textgen_llm = Bedrock(model_id = "anthropic.claude-v2",
                    client = bedrock_runtime, 
                    model_kwargs = inference_modifier 
                    )


# ## Create a LangChain custom prompt template
# 
# By creating a template for the prompt we can pass it different input variables to it on every run. This is useful when you have to generate content with different input variables that you may be fetching from a database.

# In[7]:


# Vehicle Fleet Management Code written in C++
sample_code = """
#include <iostream>
#include <string>
#include <vector>

class Vehicle {
protected:
    std::string registrationNumber;
    int milesTraveled;
    int lastMaintenanceMile;

public:
    Vehicle(std::string regNum) : registrationNumber(regNum), milesTraveled(0), lastMaintenanceMile(0) {}

    virtual void addMiles(int miles) {
        milesTraveled += miles;
    }

    virtual void performMaintenance() {
        lastMaintenanceMile = milesTraveled;
        std::cout << "Maintenance performed for vehicle: " << registrationNumber << std::endl;
    }

    virtual void checkMaintenanceDue() {
        if ((milesTraveled - lastMaintenanceMile) > 10000) {
            std::cout << "Vehicle: " << registrationNumber << " needs maintenance!" << std::endl;
        } else {
            std::cout << "No maintenance required for vehicle: " << registrationNumber << std::endl;
        }
    }

    virtual void displayDetails() = 0;

    ~Vehicle() {
        std::cout << "Destructor for Vehicle" << std::endl;
    }
};

class Truck : public Vehicle {
    int capacityInTons;

public:
    Truck(std::string regNum, int capacity) : Vehicle(regNum), capacityInTons(capacity) {}

    void displayDetails() override {
        std::cout << "Truck with Registration Number: " << registrationNumber << ", Capacity: " << capacityInTons << " tons." << std::endl;
    }
};

class Car : public Vehicle {
    std::string model;

public:
    Car(std::string regNum, std::string carModel) : Vehicle(regNum), model(carModel) {}

    void displayDetails() override {
        std::cout << "Car with Registration Number: " << registrationNumber << ", Model: " << model << "." << std::endl;
    }
};

int main() {
    std::vector<Vehicle*> fleet;

    fleet.push_back(new Truck("XYZ1234", 20));
    fleet.push_back(new Car("ABC9876", "Sedan"));

    for (auto vehicle : fleet) {
        vehicle->displayDetails();
        vehicle->addMiles(10500);
        vehicle->checkMaintenanceDue();
        vehicle->performMaintenance();
        vehicle->checkMaintenanceDue();
    }

    for (auto vehicle : fleet) {
        delete vehicle; 
    }

    return 0;
}
"""

# In[8]:


from langchain import PromptTemplate

# Create a prompt template that has multiple input variables
multi_var_prompt = PromptTemplate(
    input_variables=["code", "srcProgrammingLanguage", "targetProgrammingLanguage"], 
    template="""

Human: You will be acting as an expert software developer in {srcProgrammingLanguage} and {targetProgrammingLanguage}. 
You will tranlslate below code from {srcProgrammingLanguage} to {targetProgrammingLanguage} while following coding best practices.
<code>
{code}
</code>

Assistant: """
)

# Pass in values to the input variables
prompt = multi_var_prompt.format(code=sample_code, srcProgrammingLanguage="C++", targetProgrammingLanguage="Java")


# ### Code translation from C++ to Java

# In[9]:


response = textgen_llm(prompt)

target_code = response[response.index('\n')+1:]

print(target_code)

# ## Summary
# 
# In this example, you have learned how to translate a legacy C++ program to Java with a simple text prompt using Amazon Bedrock and langchain.
