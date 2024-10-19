#!/usr/bin/env python
# coding: utf-8

# In[4]:


# !apt-get -q update && apt-get install -y -q python3-opencv 

# In[ ]:


# %pip install -U \
#   torch==2.0.1 \
#   accelerate==0.24.1 \
#   transformers==4.35.2 \
#   torchvision==0.15.2 \
#   diffusers==0.23.1 \
#   xformers==0.0.21 \
#   bitsandbytes==0.41.1 \
#   pillow==10.0.0 \
#   opencv-python==4.8.0.76 \
#   controlnet-aux==0.0.6

# In[ ]:


!mkdir -p output/

# # Tommy

# ![](images_controlnet/tommy.png)

# In[ ]:


from diffusers import StableDiffusionControlNetPipeline
#from diffusers.utils import load_image
from diffusers.utils.testing_utils import load_image

image = load_image("images_controlnet/tommy.png")

# Render the canny edge map for this particular image
import cv2
from PIL import Image
import numpy as np

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

canny_image.save("./output/canny_image_tommy.png")

canny_image

# In[ ]:


from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel
import torch

canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
)

sd_pipe = StableDiffusionControlNetPipeline.from_pretrained(      
    "runwayml/stable-diffusion-v1-5", 
    controlnet=canny, 
)

generator = torch.manual_seed(0)

out_image = sd_pipe(
    "girl with blue tongue", 
    num_inference_steps=20, 
    generator=generator, 
    image=canny_image
).images[0]

out_image.save("./output/out_image_tommy.png")

out_image

# # Car

# ![](images_controlnet/sportscar.jpeg)

# In[8]:


from diffusers import StableDiffusionControlNetPipeline
#from diffusers.utils import load_image
from diffusers.utils.testing_utils import load_image

image = load_image("images_controlnet/sportscar.jpeg")

# Render the canny edge map for this particular image
import cv2
from PIL import Image
import numpy as np

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

canny_image.save("./output/canny_image_car.png")

canny_image

# In[9]:


from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel
import torch

canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
)

sd_pipe = StableDiffusionControlNetPipeline.from_pretrained(      
    "runwayml/stable-diffusion-v1-5", 
    controlnet=canny, 
)

generator = torch.manual_seed(0)

out_image = sd_pipe(
    prompt="metal orange colored car, complete car, color photo, outdoors in a pleasant landscape, realistic, high quality", 
    negative_prompt="cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, blurry,bad proportions", 
    num_inference_steps=20, 
    generator=generator, 
    image=canny_image
).images[0]

out_image.save("./output/out_image_car.png")

out_image

# # Meredith

# ![](images_controlnet/meredith.png)

# In[10]:


from diffusers import StableDiffusionControlNetPipeline
#from diffusers.utils import load_image
from diffusers.utils.testing_utils import load_image

image = load_image("images_controlnet/meredith.png")

# Render the canny edge map for this particular image
import cv2
from PIL import Image
import numpy as np

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

canny_image.save("./output/canny_image.png")

canny_image

# In[11]:


from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel
import torch

canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
)

sd_pipe = StableDiffusionControlNetPipeline.from_pretrained(      
    "runwayml/stable-diffusion-v1-5", 
    controlnet=canny, 
)

generator = torch.manual_seed(0)

out_image = sd_pipe(
    "disco dancer with colorful lights", 
    num_inference_steps=20, 
    generator=generator, 
    image=canny_image
).images[0]

out_image.save("./output/out_image.png")

out_image

# # Canny Image - Background
# ![](images_controlnet/landscape.png)

# In[12]:


from PIL import Image
import cv2
import numpy as np
#from diffusers.utils import load_image
from diffusers.utils.testing_utils import load_image

canny_image = load_image("images_controlnet/landscape.png")
canny_image = np.array(canny_image)
low_threshold = 100
high_threshold = 200
canny_image = cv2.Canny(canny_image, 
  low_threshold, 
  high_threshold)


# zero out middle columns of image where pose will be overlayed
zero_start = canny_image.shape[1] // 4
zero_end = zero_start + canny_image.shape[1] // 2
canny_image[:, zero_start:zero_end] = 0
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
canny_image = Image.fromarray(canny_image)

canny_image.save("./output/canny_image_masked.png")

canny_image

# # Openpose - Person

# ![](images_controlnet/tommy.png)

# In[13]:


from controlnet_aux import OpenposeDetector
#from diffusers.utils import load_image
from diffusers.utils.testing_utils import load_image

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

openpose_image = load_image("images_controlnet/tommy.png")

openpose_image = openpose(openpose_image)

openpose_image.save("./output/openpose.png")

openpose_image

# In[14]:


from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel
import torch

controls = [      
  ControlNetModel\
    .from_pretrained("lllyasviel/sd-controlnet-openpose",  
),
  ControlNetModel\
    .from_pretrained("lllyasviel/sd-controlnet-canny", 
)
]
pipe = StableDiffusionControlNetPipeline\
  .from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controls, 
)
prompt = "boy giant pig head in a fantasy landscape, best quality"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
generator = torch.Generator(device="cpu").manual_seed(1)
images = [openpose_image, canny_image]

image = pipe(
    prompt,
    images,
    num_inference_steps=20,
    generator=generator,
    negative_prompt=negative_prompt,
    controlnet_conditioning_scale=[1.0, 0.8],
).images[0]

image.save("./output/output.png")

image 

# In[ ]:



