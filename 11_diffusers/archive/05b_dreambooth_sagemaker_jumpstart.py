#!/usr/bin/env python
# coding: utf-8

# # Introduction to JumpStart - Text to Image

# ***
# Welcome to Amazon [SageMaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html)! You can use JumpStart to solve many Machine Learning tasks through one-click in SageMaker Studio, or through [SageMaker JumpStart API](https://sagemaker.readthedocs.io/en/stable/overview.html#use-prebuilt-models-with-sagemaker-jumpstart).  In this demo notebook, we demonstrate how to use the JumpStart API to generate images from text using state-of-the-art Stable Diffusion models. Furthermore, we show how to fine-tune the model to your dataset.
# 
# Stable Diffusion is a text-to-image model that enables you to create photorealistic images from just a text prompt. A diffusion model trains by learning to remove noise that was added to a real image. This de-noising process generates a realistic image. These models can also generate images from text alone by conditioning the generation process on the text. For instance, Stable Diffusion is a latent diffusion where the model learns to recognize shapes in a pure noise image and gradually brings these shapes into focus if the shapes match the words in the input text.
# 
# Training and deploying large models and running inference on models such as Stable Diffusion is often challenging and include issues such as cuda out of memory, payload size limit exceeded and so on.  JumpStart simplifies this process by providing ready-to-use scripts that have been robustly tested. Furthermore, it provides guidance on each step of the process including the recommended instance types, how to select parameters to guide image generation process, prompt engineering etc. Moreover, you can deploy and run inference on any of the 80+ Diffusion models from JumpStart without having to write any piece of your own code.
# 
# In the first part of this notebook, you will learn how to use JumpStart to generate highly realistic and artistic images of any subject, object, environment orscene. This may be as simple as an image of a cute dog or as detailed as a hyper-realistic image of a beautifully decoraded cozy kitchen by pixer in the style of Greg Rutkowski with dramatic sunset lighting and long shadows with cinematic atmosphere. This can be used to design products and build catalogs for ecommerce business needs or to generate realistic art pieces or stock images.
# 
# In the second part of this notebook, you will learn how to use JumpStart to fine-tune the Stable Diffusion model to your dataset. This can be useful when creating art, logos, custom designs, NFTs, and so on, or fun stuff such as generating custom AI images of your pets or avatars of yourself.
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
#     * [Supported Inference parameters](#2.4.-Supported-Inference-parameters)
#     * [Compressed Image Output](#2.5.-Compressed-Image-Output)
#     * [Prompt Engineering](#2.6.-Prompt-Engineering)
#     * [Negative Prompting](#2.7.-Negative-Prompting)
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
# Using JumpStart, we can perform inference on the pre-trained model, even without fine-tuning it first on a new dataset.
# ***

# ### 2.1. Select a Model
# ***
# You can continue with the default model, or can choose a different model from the dropdown generated upon running the next cell. A complete list of SageMaker pre-trained models can also be accessed at [Sagemaker pre-trained Models](https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html#).
# 
# ***

# In[4]:


# from ipywidgets import Dropdown
# from sagemaker.jumpstart.notebook_utils import list_jumpstart_models

# # Retrieves all Text-to-Image generation models.
# filter_value = "task == txt2img"
# txt2img_models = list_jumpstart_models(filter=filter_value)

# # display the model-ids in a dropdown to select a model for inference.
# model_dropdown = Dropdown(
#     options=txt2img_models,
#     value="model-txt2img-stabilityai-stable-diffusion-v2-1-base",
#     description="Select a model",
#     style={"description_width": "initial"},
#     layout={"width": "max-content"},
# )
# display(model_dropdown)

# In[7]:


# model_version="*" fetches the latest version of the model
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

# In[8]:


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

# In[9]:


import matplotlib.pyplot as plt
import numpy as np


def query(model_predictor, text):
    """Query the model predictor."""

    encoded_text = text.encode("utf-8")

    query_response = model_predictor.predict(
        encoded_text,
        {
            "ContentType": "application/x-text",
            "Accept": "application/json",
        },
    )
    return query_response


def parse_response(query_response):
    """Parse response and return generated image and the prompt"""

    response_dict = json.loads(query_response)
    return response_dict["generated_image"], response_dict["prompt"]


def display_img_and_prompt(img, prmpt):
    """Display hallucinated image."""
    plt.figure(figsize=(12, 12))
    plt.imshow(np.array(img))
    plt.axis("off")
    plt.title(prmpt)
    plt.show()

# ***
# Below, we put in some example input text. You can put in any text and the model predicts the image corresponding to that text.
# 
# ***

# In[10]:


text = "cottage in impressionist style"
query_response = query(model_predictor, text)
img, prmpt = parse_response(query_response)
display_img_and_prompt(img, prmpt)

# ### 2.4. Supported Inference parameters
# 
# ***
# This model also supports many advanced parameters while performing inference. They include:
# 
# * **prompt**: prompt to guide the image generation. Must be specified and can be a string or a list of strings.
# * **width**: width of the hallucinated image. If specified, it must be a positive integer divisible by 8.
# * **height**: height of the hallucinated image. If specified, it must be a positive integer divisible by 8.
# * **num_inference_steps**: Number of denoising steps during image generation. More steps lead to higher quality image. If specified, it must a positive integer.
# * **guidance_scale**: Higher guidance scale results in image closely related to the prompt, at the expense of image quality. If specified, it must be a float. guidance_scale<=1 is ignored.
# * **negative_prompt**: guide image generation against this prompt. If specified, it must be a string or a list of strings and used with guidance_scale. If guidance_scale is disabled, this is also disabled. Moreover, if prompt is a list of strings then negative_prompt must also be a list of strings. 
# * **num_images_per_prompt**: number of images returned per prompt. If specified it must be a positive integer. 
# * **seed**: Fix the randomized state for reproducibility. If specified, it must be an integer.
# 
# ***

# In[11]:


import json

# Training data for different models had different image sizes and it is often observed that the model performs best when the generated image
# has dimensions same as the training data dimension. For dimensions not matching the default dimensions, it may result in a black image.
# Stable Diffusion v1-4 was trained on 512x512 images and Stable Diffusion v2 was trained on 768x768 images.
payload = {
    "prompt": "astronaut on a horse",
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
    return response_dict["generated_images"], response_dict["prompt"]


query_response = query_endpoint_with_json_payload(
    model_predictor, payload, "application/json", "application/json"
)
generated_images, prompt = parse_response_multiple_images(query_response)

for img in generated_images:
    display_img_and_prompt(img, prompt)

# ### 2.5. Compressed Image Output
# 
# ---
# 
# Default response type above from an endpoint is a nested array with RGB values and if the generated image size is large, this may hit response size limit. To address this, we also support endpoint response where each image is returned as a JPEG image returned as bytes. To do this, please set `Accept = 'application/json;jpeg'`.
# 
# 
# ---

# In[ ]:


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
        model_predictor, payload, "application/json", "application/json;jpeg"
    )
    generated_images, prompt = parse_response_multiple_images(query_response)

    display_encoded_images(generated_images, title)


payload = {
    "prompt": "astronaut on a horse",
    "width": 512,
    "height": 512,
    "num_images_per_prompt": 1,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 1,
}
compressed_output_query_and_display(payload, "generated image with compressed response type")

# ### 2.6. Prompt Engineering
# ---
# Writing a good prompt can sometime be an art. It is often difficult to predict whether a certain prompt will yield a satisfactory image with a given model. However, there are certain templates that have been observed to work. Broadly, a prompt can be roughly broken down into three pieces: (i) type of image (photograph/sketch/painting etc.), (ii) description (subject/object/environment/scene etc.) and (iii) the style of the image (realistic/artistic/type of art etc.). You can change each of the three parts individually to generate variations of an image. Adjectives have been known to play a significant role in the image generation process. Also, adding more details help in the generation process.
# 
# To generate a realistic image, you can use phrases such as “a photo of”, “a photograph of”, “realistic” or “hyper realistic”. To generate images by artists you can use phrases like “by Pablo   Piccaso” or “oil painting by Rembrandt” or “landscape art by Frederic Edwin Church” or “pencil drawing by Albrecht Dürer”. You can also combine different artists as well. To generate artistic image by category, you can add the art category in the prompt such as “lion on a beach, abstract”. Some other categories include “oil painting”, “pencil drawing, “pop art”, “digital art”, “anime”, “cartoon”, “futurism”, “watercolor”, “manga” etc. You can also include details such as lighting or camera lens such as 35mm wide lens or 85mm wide lens and details about the framing (portrait/landscape/close up etc.).
# 
# Note that model generates different images even if same prompt is given multiple times. So, you can generate multiple images and select the image that suits your application best.
# 
# ---

# In[ ]:


prompts = [
    "An african woman with turban smiles at the camera, pexels contest winner, smiling young woman, face is wrapped in black scarf, a cute young woman, loosely cropped, beautiful girl, acting headshot",
    "Ancient Japanese Samurai, a person standing on a ledge in a city at night, cyberpunk art, trending on shutterstock, batman mecha, stylized cyberpunk minotaur logo, cinematic, cyberpunk",
    "Character design of a robot warrior, concept art, contest winner, diverse medical cybersuits, Football armor, triade color scheme, black shirt underneath armor, in golden armor, clothes in military armor, high resolution render, octane",
    "A croissant sitting on top of a yellow plate, a portait, trending on unsplash, sitting on a mocha-coloured table, magazine, woodfired, bakery, great composition, amber",
    "symmetry!! portrait of vanessa hudgens in the style of horizon zero dawn, machine face, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha, 8 k",
    "landscape of the beautiful city of paris rebuilt near the pacific ocean in sunny california, amazing weather, sandy beach, palm trees, splendid haussmann architecture, digital painting, highly detailed, intricate, without duplication, art by craig mullins, greg rutkwowski, concept art, matte painting, trending on artstation",
]
for prompt in prompts:
    payload = {"prompt": prompt, "width": 512, "height": 512, "seed": 1}
    compressed_output_query_and_display(payload, "generated image with detailed prompt")

# ### 2.7. Negative Prompting
# ---
# 
# Negative prompt is an important parameter while generating images using Stable Diffusion Models. It provides you an additional control over the image generation process and let you direct the model to avoid certain objects, colors, styles, attributes and more from the generated images.  
# 
# ---

# In[ ]:


prompt = "emma watson as nature magic celestial, top down pose, long hair, soft pink and white transparent cloth, space, D&D, shiny background, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, artgerm, bouguereau"
payload = {"prompt": prompt, "seed": 0}
compressed_output_query_and_display(payload, "generated image with no negative prompt")


negative_prompt = "windy"
payload = {"prompt": prompt, "negative_prompt": negative_prompt, "seed": 0}
compressed_output_query_and_display(
    payload, f"generated image with negative prompt: `{negative_prompt}`"
)

# ---
# 
# Even though, you can specify many of these concepts in the original prompt by specifying negative words “without”, “except”, “no” and “not”, Stable Diffusion models have been observed to not understand the negative words very well. Thus, you should use negative prompt parameter when tailoring the image to your use case. 
# 
# ---

# In[ ]:


prompt = "a portrait of a man without beard"
payload = {"prompt": prompt, "seed": 0}
compressed_output_query_and_display(payload, f"prompt: `{prompt}`, negative prompt: None")

prompt, negative_prompt = "a portrait of a man", "beard"
payload = {"prompt": prompt, "negative_prompt": negative_prompt, "seed": 0}
compressed_output_query_and_display(
    payload, f"prompt: `{prompt}`, negative prompt: `{negative_prompt}`"
)

# ---
# While trying to generate images, we recommend starting with prompt and progressively building negative prompt to exclude the subjects/styles that you do not want in the image.
# 
# ---
# 

# In[ ]:


prompt = "cyberpunk forest by Salvador Dali"
payload = {"prompt": prompt, "seed": 1}
compressed_output_query_and_display(payload, f"prompt: `{prompt}`, negative prompt: None")

negative_prompt = "trees, green"
payload = {"prompt": prompt, "negative_prompt": negative_prompt, "seed": 1}
compressed_output_query_and_display(
    payload, f"prompt: `{prompt}`, negative prompt: `{negative_prompt}`"
)

# ---
# Some of the helpful keywords while constructing negative prompts are: duplicate, blurry, Missing legs, mutation, morbid, deformed, malformed limbs, missing legs, bad anatomy, extra fingers, cloned face, too many fingers. 
# 
# ---

# In[ ]:


prompt = "a fantasy style portrait painting of rachel lane / alison brie / sally kellerman hybrid in the style of francois boucher oil painting unreal 5 daz. rpg portrait, extremely detailed artgerm greg rutkowski alphonse mucha"
payload = {"prompt": prompt, "seed": 1}
compressed_output_query_and_display(payload, f"No negative prompt")


negative_prompt = "duplicate"
payload = {"prompt": prompt, "negative_prompt": negative_prompt, "seed": 1}
compressed_output_query_and_display(payload, f"negative prompt: `{negative_prompt}`")

# ---
# 
# You can also use negative prompts to substitute parts of the prompt. For instance, instead of using “sharp”/“focused” in the prompt, you can use “blurry” in the negative prompt. 
# 
# Negative Prompts have been observed to be critical especially for Stable Diffusion V2 (identified by model_id `model-txt2img-stabilityai-stable-diffusion-v2`, `model-txt2img-stabilityai-stable-diffusion-v2-fp16`, `model-txt2img-stabilityai-stable-diffusion-v2-1-base`). Thus, we recommend usage of negative prompts especially when using version 2.x. To learn more about negative prompting, please see [How to use negative prompts?](https://stable-diffusion-art.com/how-to-use-negative-prompts/) and [How does negative prompt work?](https://stable-diffusion-art.com/how-negative-prompt-work/)
# 
# ---

# ### 2.8. Clean up the endpoint

# In[ ]:


# Delete the SageMaker endpoint
model_predictor.delete_model()
model_predictor.delete_endpoint()

# ## 3. Fine-tune the pre-trained model on a custom dataset
# 
# ---
# Previously, we saw how to run inference on a pre-trained model. Next, we discuss how a model can be finetuned to a custom dataset with any number of classes.
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
# The s3 path should look like `s3://bucket_name/input_directory/`. Note the trailing `/` is required.
# 
# Here is an example format of the training data.
# 
#     input_directory
#         |---instance_image_1.png
#         |---instance_image_2.png
#         |---instance_image_3.png
#         |---instance_image_4.png
#         |---instance_image_5.png
#         |---dataset_info.json
#         |---class_data_dir
#             |---class_image_1.png
#             |---class_image_2.png
#             |---class_image_3.png
#             |---class_image_4.png
# 
# **Prior preservation, instance prompt and class prompt:** Prior preservation is a technique that uses additional images of the same class that we are trying to train on.  For instance, if the training data consists of images of a particular dog, with prior preservation, we incorporate class images of generic dogs. It tries to avoid overfitting by showing images of different dogs while training for a particular dog. Tag indicating the specific dog present in instance prompt is missing in the class prompt. For instance, instance prompt may be "a photo of a riobugger cat" and class prompt may be \"a photo of a cat\". You can enable prior preservation by setting the hyper-parameter with_prior_preservation = True.
# 
# 
# We provide default datasets of cat and dogs images. Cat dataset consists of eight images (instance images corresponding to instance prompt) of a single cat with no class images. It can be downloaded from [here](https://github.com/marshmellow77/dreambooth-sm/tree/main/training-images). If using the cat dataset, try the prompt "a photo of a riobugger cat" while doing inference in the demo notebook. Dog dataset consists of 12 images of a single dog with no class images. If using the dog dataset, try the prompt "a photo of a Doppler dog" while doing inference in the demo notebook.
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

# ### 3.2. Set Training parameters
# 
# ---
# Now that we are done with all the set up that is needed, we are ready to train our stable diffusion model. To begin, let us create a [``sageMaker.estimator.Estimator``](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html) object. This estimator will launch the training job.
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

# In[ ]:


import sagemaker.metric_definitions
# Sample training data is available in this bucket
training_data_bucket = f"jumpstart-cache-prod-{aws_region}"
training_data_prefix = "training-datasets/dogs_sd_finetuning/"

training_dataset_s3_path = f"s3://{training_data_bucket}/{training_data_prefix}"

output_bucket = sess.default_bucket()
output_prefix = "jumpstart-example-sd-training"

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
        "max_jobs": 3,
        "max_parallel_jobs": 3,
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


text = "a photo of a Doppler dog with a hat"
query_response = query(finetuned_predictor, text)
img, prmpt = parse_response(query_response)
display_img_and_prompt(img, prmpt)

# All the parameters mentioned in [2.4. Supported Inference parameters](#2.4.-Supported-Inference-parameters) are supported with finetuned model as well. You may also receive compressed image output as in [2.5. Compressed Image Output](#2.5.-Compressed-Image-Output) by changing `accept`.

# ---
# Next, we delete the endpoint corresponding to the finetuned model.
# 
# ---

# In[ ]:


# Delete the SageMaker endpoint
finetuned_predictor.delete_model()
finetuned_predictor.delete_endpoint()

# ### 4. Conclusion
# ---
# In this tutorial, we learnt how to deploy a pre-trained Stable Diffusion model on SageMaker using JumpStart. We saw that Stable Diffusion models can generate highly photo-realistic images from text.  JumpStart provides both Stable Diffusion 1 and Stable Diffusion 2 and their FP16 revisions. JumpStart also provides additional 84 diffusion models which have been trained to generate images from different themes and different languages. You can deploy any of these models without writing any code of your own.  To deploy a specific model, you can select a `model_id` in the dropdown menu in [2.1. Select a Model](#2.1.-Select-a-Model).
# 
# You can tweak the image generation process by selecting the appropriate parameters during inference. Guidance on how to set these parameters is provided in [2.4. Supported Inference parameters](#2.4.-Supported-Inference-parameters). We also saw how returning a large image payload can lead to response size limit issues. JumpStart handles it by encoding the image at the endpoint and decoding it in the notebook before displaying. Finally, we saw how prompt engineering is a crucial step in generating high quality images. We discussed how to set your own prompts and saw a some examples of good prompts.
# 
# To learn more about Inference on pre-trained Stable Diffusion models, please check out the blog [Generate images from text with the stable diffusion model on Amazon SageMaker JumpStart](https://aws.amazon.com/blogs/machine-learning/generate-images-from-text-with-the-stable-diffusion-model-on-amazon-sagemaker-jumpstart/)
# 
# Although creating impressive images can find use in industries ranging from art to NFTs and beyond, today we also expect AI to be personalizable. JumpStart provides fine-tuning capability to the pre-trained models so that you can adapt the model to your own use case with as little as five training images. This can be useful when creating art, logos, custom designs, NFTs, and so on, or fun stuff such as generating custom AI images of your pets or avatars of yourself. To learn more about Stable Diffusion fine-tuning, please check out the blog [Fine-tune text-to-image Stable Diffusion models with Amazon SageMaker JumpStart](https://aws.amazon.com/blogs/machine-learning/fine-tune-text-to-image-stable-diffusion-models-with-amazon-sagemaker-jumpstart/).

# In[ ]:



