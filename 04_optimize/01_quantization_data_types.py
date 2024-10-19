#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %pip install -U torch==2.0.1

# https://pytorch.org/docs/stable/tensors.html

# In[3]:


import torch

# # Functions to convert from real to FP32, FP16, BFLOAT16, INT8

# ### Show memory footprint

# In[4]:


def show_memory_comsumption(tensor):
    memory_bytes = tensor.element_size() * tensor.numel()
    print("Tensor memory consumption:", memory_bytes, "bytes")

# ### Show binary tensor values

# In[5]:


def show_binary_tensor_values(tensor):
    
    if tensor.dtype==torch.float32:
        # Access the underlying data as 32-bit integer
        bits = tensor.view(torch.int32).item()
        binary_repr = bin(bits)
        print("Binary: ", binary_repr)      
        print("Binary representation: ", format(bits, '031b'))

    elif tensor.dtype==torch.float16:
        bits = tensor.view(torch.int16).item()
        binary_repr = bin(bits)
        print("Binary: ", binary_repr) 
        print("Binary representation: ", format(bits, '015b'))

    
    elif tensor.dtype==torch.bfloat16:
        bits = tensor.view(torch.int16).item()
        binary_repr = bin(bits)
        print("Binary: ", binary_repr) 
        print("Binary representation: ", format(bits, '015b'))


    elif tensor.dtype==torch.int8:
        bits = tensor.view(torch.int8).item()
        binary_repr = bin(bits)
        print("Binary: ", binary_repr) 
        print("Binary representation: ", format(bits, '07b'))


# ### dtype conversions

# In[6]:


def convert_to_fp32(value):
    tensor = torch.tensor(value)
    float32_tensor = tensor.float()
    print(float32_tensor.dtype)
    show_memory_comsumption(float32_tensor)
    show_binary_tensor_values(float32_tensor)
    return float32_tensor

def convert_to_fp16(value):
    tensor = torch.tensor(value)
    float16_tensor = tensor.half()
    show_memory_comsumption(float16_tensor)
    show_binary_tensor_values(float16_tensor)
    return float16_tensor

def convert_to_bfloat16(value):
    tensor = torch.tensor(value)
    bfloat16_tensor = tensor.bfloat16()
    show_memory_comsumption(bfloat16_tensor)
    show_binary_tensor_values(bfloat16_tensor)
    return bfloat16_tensor

def convert_to_int8(value):
    tensor = torch.tensor(value)
    int8_tensor = tensor.to(torch.int8)
    show_memory_comsumption(int8_tensor)
    show_binary_tensor_values(int8_tensor)
    return int8_tensor

# # Set real number

# In[7]:


#real_number = 3.141592653589793238
#real_number = 3.1415926535
#real_number = 3.141592
real_number = 3.141592
#real_number = 500.141592653589793238

# # Convert

# In[8]:


torch.set_printoptions(precision=64)

# ## FP32

# In[9]:


torch.finfo(torch.float32)

# In[10]:


float64_test = torch.tensor(real_number, dtype=torch.float64)
float32_test = torch.tensor(real_number, dtype=torch.float32)

# In[11]:


print(float64_test.item())
print(float32_test.item())

# In[12]:


float64_test

# In[13]:


float32_test

# In[14]:


# Convert to FP32
fp32_number = convert_to_fp32(real_number)
print("FP32 value:", fp32_number.item(), fp32_number)

# # Convert back to real number
# real_number_from_fp32 = float(fp32_number)
# print("Real number from FP32:", real_number_from_fp32)

# ## FP16

# In[15]:


torch.finfo(torch.float16)

# In[16]:


# Convert to FP16
fp16_number = convert_to_fp16(real_number)
print("FP16 value:", fp16_number.item(), fp16_number)

# # Convert back to real number
# real_number_from_fp16 = float(fp16_number)
# print("Real number from FP16:", real_number_from_fp16)

# ## INT8

# In[17]:


torch.iinfo(torch.int8)

# In[18]:


# Convert to INT8
int8_number = convert_to_int8(real_number)
print("INT8 value:", int8_number.item(),int8_number)

# ## BFLOAT16

# In[19]:


torch.finfo(torch.bfloat16)

# In[20]:


# Convert to BFLOAT16
bfloat16_number = convert_to_bfloat16(real_number)
print("BFLOAT16 value:", bfloat16_number.item(), bfloat16_number)

# # Convert back to real number
# real_number_from_bfloat16 = float(bfloat16_number)
# print("Real number from BFLOAT16:", real_number_from_bfloat16)
