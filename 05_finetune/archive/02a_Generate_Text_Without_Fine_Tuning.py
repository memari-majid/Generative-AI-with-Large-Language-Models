#!/usr/bin/env python
# coding: utf-8

# # Run Inference on the FLAN-T5 Model BEFORE Fine Tuning
# 
# In this lab we will now switch gears to the dialogue summarization task using generative AI instead of the data processing section. Prompt engineering is an important concept to using foundation models for text generation. We will explore how the input text can directly impact the output of the model in this notebook. Check out [this blog](https://www.amazon.science/blog/emnlp-prompt-engineering-is-the-new-feature-engineering) from Amazon Science for a quick introduction to prompt engineering.
# 
# For our specific use case, we can generate summaries with the FLAN-T5 model using zero-shot inference to see how well the base LLM performs without any fine-tuning. Check out [this blog from AWS](https://aws.amazon.com/blogs/machine-learning/zero-shot-prompting-for-the-flan-t5-foundation-model-in-amazon-sagemaker-jumpstart/) for a quick description of what zero shot learning is and why it is an important concept to the FLAN model and many more!

# <a name='1'></a>
# ## Set up Kernel and Required Dependencies

# First, check that the correct kernel is chosen.
# 
# <img src="img/kernel_set_up.png" width="300"/>
# 
# You can click on that to see and check the details of the image, kernel, and instance type.
# 
# <img src="img/w3_kernel_and_instance_type.png" width="600"/>

# In[ ]:


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

# In[ ]:


%store -r setup_dependencies_passed

# In[ ]:


try:
    setup_dependencies_passed
except NameError:
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    print("[ERROR] YOU HAVE TO RUN NOTEBOOK #01.         ")
    print("You did not install the required libraries.   ")
    print("++++++++++++++++++++++++++++++++++++++++++++++")

# In[ ]:


model_checkpoint='google/flan-t5-base'

# In[ ]:


%store model_checkpoint

# In[ ]:


%store -r model_checkpoint

# In[ ]:


huggingface_dataset_name = "knkarthick/dialogsum"

# # Load the Summarization Dataset

# In[ ]:


from datasets import load_dataset
dataset = load_dataset(huggingface_dataset_name)

# In[ ]:


example_indices = [40, 80, 160,]
print('Example Input Dialogue:')
print(dataset['test'][example_indices[0]]['dialogue'])
print()
print('Example Output Summary:')
print(dataset['test'][example_indices[0]]['summary'])

# # Load the LLM and its Tokenizer

# In[ ]:


from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# In[ ]:


from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# # Explore What Happens Without Prompt Engineering
# 
# Without any prompt engineering (changes to the dialogue text), you can see below how the model is unsure of what task it is supposed to accomplish, so it tries to make up the next sentence in the dialogue. Not a bad guess from the model, but it isn't the task we want the model to perform in this situation.

# In[ ]:


dialogue = dataset['test'][example_indices[0]]['dialogue']
summary = dataset['test'][example_indices[0]]['summary']

inputs = tokenizer(dialogue, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"], 
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)
print(f'INPUT PROMPT:\n{dialogue}\n')
print(f'MODEL GENERATION:\n{output}')

# In[ ]:


dialogue = dataset['test'][example_indices[1]]['dialogue']
summary = dataset['test'][example_indices[1]]['summary']

inputs = tokenizer(dialogue, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"], 
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)
print(f'INPUT PROMPT:\n{dialogue}\n')
print(f'MODEL GENERATION:\n{output}')

# # Add an Text Prompt to the Model Input
# 
# Adding the "summary:" command to the end seems to really help the model understand what it is supposed to do now. Notice how the model still does not pick up on the nuance of the conversations though. This is what we will hope to solve via fine-tuning.

# In[ ]:


start_prompt = 'Summarize the following conversation.\n'
end_prompt = '\n\nSummary: '
dialogue = dataset['test'][example_indices[0]]['dialogue']
summary = dataset['test'][example_indices[0]]['summary']
prompt = f'{start_prompt}{dialogue}{end_prompt}'

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"], 
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)
print(f'INPUT PROMPT:\n{prompt}\n')
print(f'MODEL GENERATION:\n{output}\n')
print(f'BASELINE SUMMARY:\n{summary}')

# In[ ]:


start_prompt = 'Summarize the following conversation.\n'
end_prompt = '\n\nSummary: '
dialogue = dataset['test'][example_indices[1]]['dialogue']
summary = dataset['test'][example_indices[1]]['summary']
prompt = f'{start_prompt}{dialogue}{end_prompt}'

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"], 
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)
print(f'INPUT PROMPT:\n{prompt}\n')
print(f'MODEL GENERATION:\n{output}\n')
print(f'BASELINE SUMMARY:\n{summary}')

# # Try Out a Different Prompt Template From FLAN
# 
# The next natural thing to check is if the text at the start and end of the dialogue are optimized to the task at hand. FLAN has many prompt templates that are published for certain tasks [here](https://github.com/google-research/FLAN/tree/main/flan/v2). As you can see below, selecting one of the pre-built FLAN prompts does help in the second example, but the first example still struggles to pick up on the nuance of the conversation.

# In[ ]:


start_prompt = 'Dialogue:\n'
end_prompt = '\nWhat was going on?'
dialogue = dataset['test'][example_indices[0]]['dialogue']
summary = dataset['test'][example_indices[0]]['summary']
prompt = f'{start_prompt}{dialogue}{end_prompt}'

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"], 
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)
print(f'INPUT PROMPT:\n{prompt}\n')
print(f'MODEL GENERATION:\n{output}\n')
print(f'BASELINE SUMMARY:\n{summary}')

# In[ ]:


start_prompt = 'Dialogue:\n'
end_prompt = '\nWhat was going on?'
dialogue = dataset['test'][example_indices[1]]['dialogue']
summary = dataset['test'][example_indices[1]]['summary']
prompt = f'{start_prompt}{dialogue}{end_prompt}'

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"], 
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)
print(f'INPUT PROMPT:\n{prompt}\n')
print(f'MODEL GENERATION:\n{output}\n')
print(f'BASELINE SUMMARY:\n{summary}')

# # Use few-shot Inference
# 
# Few shot inference is the practice of providing an LLM examples of what outputs should look like for a given task. You can read more about it in [this blog from HuggingFace](https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api) which goes over why is a useful tool and how it can be used!
# 
# In our example, you can see how providing 2 examples to the model provides the model more information and qualitatively improves the summary in the example below.

# In[ ]:


start_prompt = 'Dialogue:\n'
end_prompt = '\nWhat was going on? '
stop_sequence = '\n\n\n'

# In[ ]:


def make_prompt(num_shots):
    prompt = ''
    for i in range(num_shots + 1):
        if i == num_shots:
            dialogue = dataset['test'][example_indices[0]]['dialogue']
            summary = dataset['test'][example_indices[0]]['summary']
            prompt = prompt + f'{start_prompt}{dialogue}{end_prompt}'
        else:
            dialogue = dataset['test'][example_indices[i+1]]['dialogue']
            summary = dataset['test'][example_indices[i+1]]['summary']
            prompt = prompt + f'{start_prompt}{dialogue}{end_prompt}{summary}\n{stop_sequence}\n'
    return prompt

# In[ ]:


few_shot_prompt = make_prompt(2)
print(few_shot_prompt)

# In[ ]:


inputs = tokenizer(few_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)
print(f'FEW SHOT RESPONSE: {output}')
summary = dataset['test'][example_indices[0]]['summary']
print(f'EXPECTED RESPONSE: {summary}')

# # Conclusion
# 
# As you can see, prompt engineering can take us a long way for this use case, but there are some limitations. Next, we will start to explore how you can use fine-tuning to help your LLM to understand a particular use case in better depth!

# # Release Resources

# In[ ]:


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
