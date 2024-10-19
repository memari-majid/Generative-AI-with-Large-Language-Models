#!/usr/bin/env python
# coding: utf-8

# # Fine-tune Stable Diffusion using Dreambooth

# ***
# 
# Note:  This notebook has been adapted from the notebook provided by Amazon [SageMaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html)! You can use JumpStart to solve many Machine Learning tasks through one-click in SageMaker Studio, or through [SageMaker JumpStart API](https://sagemaker.readthedocs.io/en/stable/overview.html#use-prebuilt-models-with-sagemaker-jumpstart).  
# 
# In this notebook, we show how to fine-tune a Stable Diffusion model using your own dataset.  In this case, we'll focus on fine-tuning using a few images of Molly dog as a puppy! 
# 
# Stable Diffusion is a text-to-image model that enables you to create photorealistic images from just a text prompt. A diffusion model trains by learning to remove noise that was added to a real image. This de-noising process generates a realistic image. These models can also generate images from text alone by conditioning the generation process on the text. For instance, Stable Diffusion is a latent diffusion where the model learns to recognize shapes in a pure noise image and gradually brings these shapes into focus if the shapes match the words in the input text.
# 
# 
# Model license: By using this model, you agree to the [CreativeML Open RAIL-M++ license](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL).
# 
# ***

# 1. [Set Up](#1.-Set-Up)
# 2. [Run inference on the pre-trained model](#2.-Run-inference-on-the-pre-trained-model)
#     * [Select a model](#2.1.-Select-a-Model)
#     * [Retrieve JumpStart Artifacts & Deploy an Endpoint](#2.2.-Retrieve-JumpStart-Artifacts-&-Deploy-an-Endpoint)
#     * [Query endpoint and parse response](#2.3.-Query-endpoint-and-parse-response)
#     * [Clean up the endpoint](#2.8.-Clean-up-the-endpoint)
# 
# 3. [Fine-tune the pre-trained model on a custom dataset](#3.-Fine-tune-the-pre-trained-model-on-a-custom-dataset)
#     * [Retrieve Training Artifacts](#3.1.-Retrieve-Training-Artifacts)
#     * [Set Training parameters](#3.2.-Set-Training-parameters)
#     * [Start Training](#3.3.-Start-Training)
#     * [Deploy and run inference on the fine-tuned model](#3.4.-Deploy-and-run-inference-on-the-fine-tuned-model)
# 
# 4. [Conclusion](#4.-Conclusion)
# 

# Note: This notebook was tested on ml.t3.medium instance in Amazon SageMaker Studio with Python 3 (Data Science) kernel and in Amazon SageMaker Notebook instance with conda_python3 kernel.
# 
# Note: To deploy the pre-trained or fine-tuned model, you can use `ml.p3.2xlarge` or `ml.g4dn.2xlarge` instance types. If `ml.g5.2xlarge` is available in your region, we recommend using that instance type for deployment. For fine-tuning the model on your dataset, you need `ml.g4dn.2xlarge` instance type available in your account.

# ### 1. Set Up

# ***
# 
# Before executing the notebook, there are some initial steps required for set up. This notebook requires ipywidgets and latest version of sagemaker.
# 
# 
# ***

# In[2]:


!pip install ipywidgets==7.0.0 --quiet
!pip install --upgrade sagemaker

# #### Permissions and environment variables
# 
# ***
# To host on Amazon SageMaker, we need to set up and authenticate the use of AWS services. Here, we use the execution role associated with the current notebook as the AWS account role with SageMaker access.
# 
# ***

# In[3]:


import sagemaker, boto3, json
from sagemaker import get_execution_role

aws_role = get_execution_role()
aws_region = boto3.Session().region_name
sess = sagemaker.Session()

# ## 2. Run inference on the pre-trained model
# 
# ***
# 
# Using JumpStart, we can perform inference on the pre-trained model, even without fine-tuning it first on a new dataset.  We'll use this to demonstrate that the base foundation model doesn't know yet about Molly as Molly was not included in the training data used to pre-train Stable Diffusion.
# ***

# ### 2.1. Select a Model
# ***
# You can continue with the default model, or can choose a different model from the dropdown generated upon running the next cell. A complete list of SageMaker pre-trained models can also be accessed at [Sagemaker pre-trained Models](https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html#).
# 
# ***

# In[4]:


#model_id, model_version = "model-imagegeneration-stabilityai-stable-diffusion-xl-base-1-0", "*"
model_id, model_version = "model-imagegeneration-stabilityai-stable-diffusion-v2-1", "*"

# ### 2.2. Retrieve JumpStart Artifacts & Deploy an Endpoint
# 
# ***
# 
# Using JumpStart, we can perform inference on the pre-trained model, even without fine-tuning it first on a new dataset. We start by retrieving the `deploy_image_uri`, and `model_uri` for the pre-trained model. To host the pre-trained model, we create an instance of [sagemaker.model.Model](https://sagemaker.readthedocs.io/en/stable/api/inference/model.html) and deploy it.
# 
# ### This may take up to 10 minutes. Please do not kill the kernel while you wait.
# 
# While you wait, you can checkout the [Generate images from text with the stable diffusion model on Amazon SageMaker JumpStart](https://aws.amazon.com/blogs/machine-learning/generate-images-from-text-with-the-stable-diffusion-model-on-amazon-sagemaker-jumpstart/) blog to learn more about Stable Diffusion model and JumpStart.
# 
# 
# ***

# In[5]:


from sagemaker import image_uris, model_uris, script_uris, hyperparameters, instance_types
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.utils import name_from_base


endpoint_name = name_from_base(f"jumpstart-example-infer-{model_id}")

# Please use ml.g5.24xlarge instance type if it is available in your region. ml.g5.24xlarge has 24GB GPU compared to 16GB in ml.p3.2xlarge and supports generation of larger and better quality images.
inference_instance_type = instance_types.retrieve_default(
    region=None,
    model_id=model_id,
    model_version=model_version,
    scope="inference"
)

# Retrieve the inference docker container uri. This is the base HuggingFace container image for the default model above.
deploy_image_uri = image_uris.retrieve(
    region=None,
    framework=None,  # automatically inferred from model_id
    image_scope="inference",
    model_id=model_id,
    model_version=model_version,
    instance_type=inference_instance_type,
)

# Retrieve the model uri. This includes the pre-trained model and parameters as well as the inference scripts.
# This includes all dependencies and scripts for model loading, inference handling etc..
model_uri = model_uris.retrieve(
    model_id=model_id, model_version=model_version, model_scope="inference"
)

# To increase the maximum response size (in bytes) from the endpoint.
env = {
    "MMS_MAX_RESPONSE_SIZE": "20000000",
}

# Create the SageMaker model instance
model = Model(
    image_uri=deploy_image_uri,
    model_data=model_uri,
    role=aws_role,
    predictor_cls=Predictor,
    name=endpoint_name,
    env=env,
)

# Deploy the Model. Note that we need to pass Predictor class when we deploy model through Model class,
# for being able to run inference through the sagemaker API.
model_predictor = model.deploy(
    initial_instance_count=1,
    instance_type=inference_instance_type,
    predictor_cls=Predictor,
    endpoint_name=endpoint_name,
)

# ### 2.3. Query endpoint and parse response
# 
# ***
# Input to the endpoint is any string of text converted to json and encoded in `utf-8` format. Output of the endpoint is a `json` with generated text.
# 
# ***

# In[6]:


import numpy as np

def display_img_and_prompt(img, prmpt):
    print(img)
    plt.figure(figsize=(12, 12))
    plt.imshow(np.array(img))
    plt.axis("off")
    plt.title(prmpt)
    plt.show()

# ***
# Below, we put in some example input text. You can put in any text and the model predicts the image corresponding to that text.  Let's first have it generate a picture of a dog (any dog) drinking a mai tai on the beach.
# 
# ***

# In[7]:


# text = "dog drinking mai tai on the beach"
# query_response = query(model_predictor, text)
# img, prmpt = parse_response(query_response)
# display_img_and_prompt(img, prmpt)

# As you can see, Stable Diffusion is able to generate the image using the class of dog and the additional context provided but let's see if it can generate an image of my dog Molly on the beach...

# In[8]:


# text = "my dog molly on the beach"
# query_response = query(model_predictor, text)
# img, prmpt = parse_response(query_response)
# display_img_and_prompt(img, prmpt)

# In[9]:


payload = {
    "text_prompts": [{"text": "my dog molly on the beach"}],
    "width": 512,
    "height": 512,
    "num_images_per_prompt": 1,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 1,
}


def query_endpoint_with_json_payload(model_predictor, payload, content_type, accept):
    """Query the model predictor with json payload."""

    encoded_payload = json.dumps(payload).encode("utf-8")

    query_response = model_predictor.predict(
        encoded_payload,
        {
            "ContentType": content_type,
            "Accept": accept,
        },
    )
    return query_response


def parse_response_multiple_images(query_response):
    """Parse response and return generated image and the prompt"""

    response_dict = json.loads(query_response)
#    print(response_dict)
    return response_dict["generated_image"], response_dict["prompt"]

# In[10]:


from PIL import Image
from io import BytesIO
import base64
import json


def display_encoded_images(generated_images, title):
    """Decode the images and convert to RGB format and display

    Args:
    generated_images: are a list of jpeg images as bytes with b64 encoding.
    """

    for generated_image in generated_images:
        generated_image_decoded = BytesIO(base64.b64decode(generated_image.encode()))
        generated_image_rgb = Image.open(generated_image_decoded).convert("RGB")
        display_img_and_prompt(generated_image_rgb, title)


def compressed_output_query_and_display(payload, title):
    query_response = query_endpoint_with_json_payload(
        model_predictor, payload, "application/json", "application/json"
    )
    generated_images, prompt = parse_response_multiple_images(query_response)

    display_encoded_images(generated_images, title)


payload = {
    "text_prompts": [{"text": "my dog molly on the beach"}],
    "width": 512,
    "height": 512,
    "num_images_per_prompt": 1,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 1,
}
compressed_output_query_and_display(payload, "generated image with compressed response type")

# In[ ]:


query_response = query_endpoint_with_json_payload(
    model_predictor, payload, "application/json", "application/json"
)
generated_images, prompt = parse_response_multiple_images(query_response)

for img in generated_images:
    display_img_and_prompt(img, prompt)

# As cute as that dog is, it's not my dog Molly because Stable Diffusion doesn't know about Molly yet.   To allow Stable Diffusion to create images of Molly in new exciting environments, we'll need to fine-tune in the steps covered in the next section. 
# 
# But first, let's clean up the endpoint to avoid unnecessary cost. 

# ### 2.8. Clean up the endpoint

# In[ ]:


# # Delete the SageMaker endpoint
# model_predictor.delete_model()
# model_predictor.delete_endpoint()

# ## 3. Fine-tune the pre-trained model on a custom dataset
# 
# ---
# Previously, we saw how to run inference on a pre-trained model. Next, we discuss how a model can be finetuned to a custom dataset with any number of classes.  In this case, we'll fine-tune using a custom dataset with a few images of Molly. 
# 
# The model can be fine-tuned to any dataset of images. It works very well even with as little as five training images.
# 
# The fine-tuning script is built on the script from [dreambooth](https://dreambooth.github.io/). The model returned by fine-tuning can be further deployed for inference. Below are the instructions for how the training data should be formatted.
# 
# - **Input:** A directory containing the instance images, `dataset_info.json` and (optional) directory `class_data_dir`.
#   - Images may be of `.png` or `.jpg` or `.jpeg` format.
#   - `dataset_info.json` file must be of the format {'instance_prompt':<<instance_prompt>>,'class_prompt':<<class_prompt>>}.
#   - If with_prior_preservation = False, you may choose to ignore 'class_prompt'.
#   - `class_data_dir` directory must have class images. If with_prior_preservation = True and class_data_dir is not present or there are not enough images already present in class_data_dir, additional images will be sampled with class_prompt.
# - **Output:** A trained model that can be deployed for inference.
# 
# **Prior preservation, instance prompt and class prompt:** Prior preservation is a technique that uses additional images of the same class that we are trying to train on.  For instance, in this case we are fine-tuning with images of a specific dog, Molly,.  With prior preservation, we incorporate class images of generic dogs to avoid overfitting by showing images of different dogs while training for a particular dog. We then have a tag indicating the specific dog present in instance prompt is missing in the class prompt. For instance, instance prompt may be "photo of a molly dog" and class prompt may be \"a photo of a dog\". You can enable prior preservation by setting the hyper-parameter with_prior_preservation = True.
# 
# 
# License: [MIT](https://github.com/marshmellow77/dreambooth-sm/blob/main/LICENSE).

# ### 3.1. Retrieve Training Artifacts
# 
# ---
# Here, we retrieve the training docker container, the training algorithm source, and the pre-trained base model. Note that model_version="*" fetches the latest model.
# 
# ---

# In[ ]:


from sagemaker import image_uris, model_uris, script_uris

# Currently, not all the stable diffusion models in jumpstart support finetuning. Thus, we manually select a model
# which supports finetuning.
train_model_id, train_model_version, train_scope = (
    "model-txt2img-stabilityai-stable-diffusion-v2-1-base",
    "*",
    "training",
)

# Tested with ml.g4dn.2xlarge (16GB GPU memory) and ml.g5.2xlarge (24GB GPU memory) instances. Other instances may work as well.
# If ml.g5.2xlarge instance type is available, please change the following instance type to speed up training.
training_instance_type = instance_types.retrieve_default(
    region=None,
    model_id=train_model_id,
    model_version=train_model_version,
    scope=train_scope
)

# Retrieve the docker image
train_image_uri = image_uris.retrieve(
    region=None,
    framework=None,  # automatically inferred from model_id
    model_id=train_model_id,
    model_version=train_model_version,
    image_scope=train_scope,
    instance_type=training_instance_type,
)

# Retrieve the training script. This contains all the necessary files including data processing, model training etc.
train_source_uri = script_uris.retrieve(
    model_id=train_model_id, model_version=train_model_version, script_scope=train_scope
)
# Retrieve the pre-trained model tarball to further fine-tune
train_model_uri = model_uris.retrieve(
    model_id=train_model_id, model_version=train_model_version, model_scope=train_scope
)

print(training_instance_type)

# ### 3.2. Set Training parameters
# 
# ---
# Now that we are done with all the set up that is needed, we are ready to train our stable diffusion model. To begin, let us create a[``sageMaker.estimator.Estimator``](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html) object. This estimator will launch the training job.
# 
# 
# There are two kinds of parameters that need to be set for training. The first one are the parameters for the training job. These include:
# - Training data path. This is S3 folder in which the input data is stored.
# - Output path: This the s3 folder in which the training output is stored.
# - Training instance type: This indicates the type of machine on which to run the training. We defined the training instance type above to fetch the correct train_image_uri.
# - Metric definitions: A list of dictionaries that define the metrics used to evaluate the training job. Default values are retrieved from the SageMaker SDK.
# 
# The second set of parameters are algorithm specific training hyper-parameters.
# 
# ---

# First, upload the training images to S3.

# In[ ]:


default_bucket = sess.default_bucket()

training_data_bucket = default_bucket
training_data_prefix = "stable-diffusion/training_images/"

training_dataset_s3_path = f"s3://{training_data_bucket}/{training_data_prefix}"
print(training_dataset_s3_path)

# In[ ]:


!aws s3 cp images_molly_sagemaker_jumpstart {training_dataset_s3_path}/ --recursive

# In[ ]:


import sagemaker.metric_definitions

output_bucket = sess.default_bucket()
output_prefix = "jumpstart-stable-diffusion-training"

# Retrieve the default metric definitions to emit to CloudWatch Logs\n",
metric_definitions = sagemaker.metric_definitions.retrieve_default(
    model_id=train_model_id, model_version=train_model_version,
)

s3_output_location = f"s3://{output_bucket}/{output_prefix}/output"


# ---
# For algorithm specific hyper-parameters, we start by fetching python dictionary of the training hyper-parameters that the algorithm accepts with their default values. This can then be overridden to custom values.
# 
# ---

# In[ ]:


from sagemaker import hyperparameters

# Retrieve the default hyper-parameters for fine-tuning the model
hyperparameters = hyperparameters.retrieve_default(
    model_id=train_model_id, model_version=train_model_version
)

# [Optional] Override default hyperparameters with custom values
hyperparameters["max_steps"] = "400"
print(hyperparameters)

# ---
# If setting `with_prior_preservation=True`, please use ml.g5.2xlarge instance type as more memory is required to generate class images. Currently, training on ml.g4dn.2xlarge instance type run into CUDA out of memory issue when setting `with_prior_preservation=True`.
# 
# ---

# ### 3.3. Train with Automatic Model Tuning
# ---
# 
# Amazon SageMaker automatic model tuning, also known as hyperparameter tuning, finds the best version of a model by running many training jobs on your dataset using the algorithm and ranges of hyperparameters that you specify. It then chooses the hyperparameter values that result in a model that performs the best, as measured by a metric that you choose. We will use a HyperparameterTuner object to interact with Amazon SageMaker hyperparameter tuning APIs. Here we tune two hyper-parameters `learning_rate` and `max_steps`.
# 
# ---

# In[ ]:


from sagemaker.tuner import IntegerParameter
from sagemaker.tuner import ContinuousParameter
from sagemaker.tuner import HyperparameterTuner

use_amt = True

hyperparameter_ranges = {
    "learning_rate": ContinuousParameter(1e-7, 3e-6, "Linear"),
    "max_steps": IntegerParameter(50, 400, "Linear"),
}

# ### 3.4. Start Training
# ---
# 
# We start by creating the estimator object with all the required assets and then launch the training job. 
# 
# ---

# In[ ]:


from sagemaker.estimator import Estimator
from sagemaker.utils import name_from_base
from sagemaker.tuner import HyperparameterTuner

training_job_name = name_from_base(f"jumpstart-example-{train_model_id}-transfer-learning")

# Create SageMaker Estimator instance
sd_estimator = Estimator(
    role=aws_role,
    image_uri=train_image_uri,
    source_dir=train_source_uri,
    model_uri=train_model_uri,
    entry_point="transfer_learning.py",  # Entry-point file in source_dir and present in train_source_uri.
    instance_count=1,
    instance_type=training_instance_type,
    max_run=360000,
    metric_definitions=metric_definitions,
    hyperparameters=hyperparameters,
    output_path=s3_output_location,
    base_job_name=training_job_name,
)


if use_amt:
    # Let estimator emit fid_score metric to AMT
    sd_estimator.set_hyperparameters(compute_fid="True")
    tuner_parameters = {
        "estimator": sd_estimator,
        "metric_definitions": [{"Name": "fid_score", "Regex": "fid_score=([-+]?\\d\\.?\\d*)"}],
        "objective_metric_name": "fid_score",
        "objective_type": "Minimize",
        "hyperparameter_ranges": hyperparameter_ranges,
        "max_jobs": 2,
        "max_parallel_jobs": 2,
        "strategy": "Bayesian",
        "base_tuning_job_name": training_job_name,
    }

    tuner = HyperparameterTuner(**tuner_parameters)
    tuner.fit({"training": training_dataset_s3_path}, logs=True)
else:
    # Launch a SageMaker Training job by passing s3 path of the training data
    sd_estimator.fit({"training": training_dataset_s3_path}, logs=True)

# If you set `max_jobs=9` and `max_parallel_jobs=3` above, it takes around 60 mins on the default dataset when using automatic model tuning. If more quota for the training instance type is available, please increase the `max_parallel_jobs` to speed up the tuning.

# ### 3.5. Deploy and run inference on the fine-tuned model
# 
# ---
# 
# A trained model does nothing on its own. We now want to use the model to perform inference. For this example, that means predicting the bounding boxes of an image. We follow the same steps as in [2. Run inference on the pre-trained model](#2.-Run-inference-on-the-pre-trained-model). We start by retrieving the jumpstart artifacts for deploying an endpoint. However, instead of base_predictor, we  deploy the `od_estimator` that we fine-tuned.
# 
# ---

# In[ ]:


inference_instance_type = instance_types.retrieve_default(
    region=None,
    model_id=train_model_id,
    model_version=train_model_version,
    scope="inference"
)

# Retrieve the inference docker container uri
deploy_image_uri = image_uris.retrieve(
    region=None,
    framework=None,  # automatically inferred from model_id
    image_scope="inference",
    model_id=train_model_id,
    model_version=train_model_version,
    instance_type=inference_instance_type,
)

endpoint_name = name_from_base(f"jumpstart-example-FT-{train_model_id}-")

# Use the estimator from the previous step to deploy to a SageMaker endpoint
finetuned_predictor = (tuner if use_amt else sd_estimator).deploy(
    initial_instance_count=1,
    instance_type=inference_instance_type,
    image_uri=deploy_image_uri,
    endpoint_name=endpoint_name,
)

# Next, we query the finetuned model, parse the response and display the generated image. Functions for these are implemented in sections [2.3. Query endpoint and parse response](#2.3.-Query-endpoint-and-parse-response). Please execute those cells.

# In[ ]:


text = "a photo of Molly dog on the beach"
query_response = query(finetuned_predictor, text)
img, prmpt = parse_response(query_response)
display_img_and_prompt(img, prmpt)

# In[ ]:


text = "create a photo of Molly dog in the style of Vincent Van Gogh"
query_response = query(finetuned_predictor, text)
img, prmpt = parse_response(query_response)
display_img_and_prompt(img, prmpt)

# In[ ]:


text = "create a photo of Molly dog in the style of Pierre-Auguste Renoir"
query_response = query(finetuned_predictor, text)
img, prmpt = parse_response(query_response)
display_img_and_prompt(img, prmpt)

# In[ ]:


text = "back of molly dog head"
query_response = query(finetuned_predictor, text)
img, prmpt = parse_response(query_response)
display_img_and_prompt(img, prmpt)

# In[ ]:


text = "molly dog with sunglasses"
query_response = query(finetuned_predictor, text)
img, prmpt = parse_response(query_response)
display_img_and_prompt(img, prmpt)

# In[ ]:


text = "molly dog with a hat"
query_response = query(finetuned_predictor, text)
img, prmpt = parse_response(query_response)
display_img_and_prompt(img, prmpt)

# All the inference parameters previously learned with finetuned model as well. You may also receive compressed image output by changing `accept`.
# 
# 
# Congratulations! You just fine-tuned a Stable Diffusion model.  Feel free to explore more by changing the training images and experimenting with various hyperparameters. 

# ---
# Next, we delete the endpoint corresponding to the finetuned model to avoid unnecessary cost.
# 
# ---

# In[ ]:


# # Delete the SageMaker endpoint
# finetuned_predictor.delete_model()
# finetuned_predictor.delete_endpoint()

# ### 4. Conclusion
# ---
# In this tutorial, you learned how a pre-trained Stable Diffusion model is unlikely to be able to create customized images of specific items such as your own dog because it's not in the data used to pre-train.  However, you saw that you can easily customize Stable Diffusion with a few images to allow it to create fun new images of specific subjects such as Molly dog.  You also learned you can deploy and fine-tune any of these models without writing any code of your own by using Amazon SageMaker JumpStart.

# In[ ]:



