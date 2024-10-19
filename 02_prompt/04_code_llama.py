#!/usr/bin/env python
# coding: utf-8

# # Code Llama

# Here are the names of the Code Llama models provided by Together.ai:

# - ```togethercomputer/CodeLlama-7b```
# - ```togethercomputer/CodeLlama-13b```
# - ```togethercomputer/CodeLlama-34b```
# - ```togethercomputer/CodeLlama-7b-Python```
# - ```togethercomputer/CodeLlama-13b-Python```
# - ```togethercomputer/CodeLlama-34b-Python```
# - ```togethercomputer/CodeLlama-7b-Instruct```
# - ```togethercomputer/CodeLlama-13b-Instruct```
# - ```togethercomputer/CodeLlama-34b-Instruct```

# ### Import helper functions
# 
# - You can examine the code_llama helper function using the menu above and selections File -> Open -> utils.py.
# - By default, the `code_llama` functions uses the CodeLlama-7b-Instruct model.

# In[ ]:


from utils import llama, code_llama

# ### Writing code to solve a math problem
# 
# Lists of daily minimum and maximum temperatures:

# In[ ]:


temp_min = [42, 52, 47, 47, 53, 48, 47, 53, 55, 56, 57, 50, 48, 45]
temp_max = [55, 57, 59, 59, 58, 62, 65, 65, 64, 63, 60, 60, 62, 62]

# - Ask the Llama 7B model to determine the day with the lowest temperature.

# In[ ]:


prompt = f"""
Below is the 14 day temperature forecast in fahrenheit degree:
14-day low temperatures: {temp_min}
14-day high temperatures: {temp_max}
Which day has the lowest temperature?
"""

response = llama(prompt)
print(response)

# - Ask Code Llama to write a python function to determine the minimum temperature.

# In[ ]:


prompt_2 = f"""
Write Python code that can calculate
the minimum of the list temp_min
and the maximum of the list temp_max
"""
response_2 = code_llama(prompt_2)
print(response_2)

# - Use the function on the temperature lists above.

# In[ ]:


def get_min_max(temp_min, temp_max):
    return min(temp_min), max(temp_max)

# In[ ]:


temp_min = [42, 52, 47, 47, 53, 48, 47, 53, 55, 56, 57, 50, 48, 45]
temp_max = [55, 57, 59, 59, 58, 62, 65, 65, 64, 63, 60, 60, 62, 62]

results = get_min_max(temp_min, temp_max)
print(results)

# ### Code in-filling
# 
# - Use Code Llama to fill in partially completed code.
# - Notice the `[INST]` and `[/INST]` tags that have been added to the prompt.

# In[ ]:


prompt = """
def star_rating(n):
'''
  This function returns a rating given the number n,
  where n is an integers from 1 to 5.
'''

    if n == 1:
        rating="poor"
    <FILL>
    elif n == 5:
        rating="excellent"

    return rating
"""

response = code_llama(prompt,
                      verbose=True)


# In[ ]:


print(response)

# ### Write code to calculate the nth Fibonacci number
# 
# Here is the Fibonacci sequence:

# In[ ]:


# 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610...


# Each number (after the starting 0 and 1) is equal to the sum of the two numbers that precede it.

# #### Use Code Llama to write a Fibonacci number
# - Write a natural language prompt that asks the model to write code.

# In[ ]:


prompt = """
Provide a function that calculates the n-th fibonacci number.
"""

response = code_llama(prompt, verbose=True)
print(response)

# ### Make the code more efficient
# 
# - Ask Code Llama to critique its initial response.

# In[ ]:


code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""

prompt_1 = f"""
For the following code: {code}
Is this implementation efficient?
Please explain.
"""
response_1 = code_llama(prompt_1, verbose=True)


# In[ ]:


print(response_1)

# In[ ]:




# ### Compare the original and more efficient implementations

# In[ ]:


def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# In[ ]:




# In[ ]:


def fibonacci_fast(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a


# In[ ]:




# #### Compare the runtimes of the two functions
# - Start by asking Code Llama to write Python code that calculates how long a piece of code takes to execute:

# In[ ]:


prompt = f"""
Provide sample code that calculates the runtime \
of a Python function call.
"""

response = code_llama(prompt, verbose=True)
print (response)

# Let's use the first suggestion from Code Llama to calcuate the run time.
# 
#     Here is an example of how you can calculate the runtime of a Python function call using the `time` module:
#     ```
#     import time
#     
#     def my_function():
#         # do something
#         pass
#     
#     start_time = time.time()
#     my_function()
#     end_time = time.time()
#     
#     print("Runtime:", end_time - start_time)
#     ```
# 

# #### Run the original Fibonacci code
# - This will take approximately 45 seconds.
# - The video has been edited so you don't have to wait for the code to exectute.

# In[ ]:


import time
n=40
start_time = time.time()
fibonacci(n) # note, we recommend keeping this number <=40
end_time = time.time()
print(f"recursive fibonacci({n}) ")
print(f"runtime in seconds: {end_time-start_time}")

# In[ ]:




# #### Run the efficient implementation

# In[ ]:


import time
n=40
start_time = time.time()
fibonacci_fast(n) # note, we recommend keeping this number <=40
end_time = time.time()
print(f"non-recursive fibonacci({n}) ")
print(f"runtime in seconds: {end_time-start_time}")

# In[ ]:




# ### Code Llama can take in longer text
# 
# - Code Llama models can handle much larger input text than the Llama Chat models - more than 20,000 characters.
# - The size of the input text is known as the **context window**.

# #### Response from Llama 2 7B Chat model
# - The following code will return an error because the sum of the input and output tokens is larger than the limit of the model.
# - You can revisit L2 for more details.

# In[ ]:


with open("TheVelveteenRabbit.txt", 'r', encoding='utf-8') as file:
    text = file.read()

prompt=f"""
Give me a summary of the following text in 50 words:\n\n 
{text}
"""

# Ask the 7B model to respond
response = llama(prompt)
print(response)

# In[ ]:




# #### Response from Code Llama 7B Instruct model

# In[ ]:


from utils import llama
with open("TheVelveteenRabbit.txt", 'r', encoding='utf-8') as file:
    text = file.read()

prompt=f"""
Give me a summary of the following text in 50 words:\n\n 
{text}
"""
response = code_llama(prompt)
print(response)


# In[ ]:




# ### Thoughts on Code Llama's summarization performance
# 
# Note that while the Code Llama model could handle the longer text, the output here isn't that great - the response is very repetitive.
# - Code Llama's primary skill is writing code.
# - Experiment to see if you can prompt the Code Llama model to improve its output.
# - You may need to trade off performance and input text size depending on your task.
# - You could ask Llama 2 70B chat to help you evaluate how well the Code Llama model is doing!

# In[ ]:



