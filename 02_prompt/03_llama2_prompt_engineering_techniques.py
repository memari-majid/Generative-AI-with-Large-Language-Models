#!/usr/bin/env python
# coding: utf-8

# # Lesson 4

# ### Import helper function

# In[ ]:


from utils import llama, llama_chat

# ### In-Context Learning
# 
# #### Standard prompt with instruction
# - So far, you have been stating the instruction explicitly in the prompt:

# In[ ]:


prompt = """
What is the sentiment of:
Hi Amit, thanks for the thoughtful birthday card!
"""
response = llama(prompt)
print(response)

# ### Zero-shot Prompting
# - Here is an example of zero-shot prompting.
# - You are prompting the model to see if it can infer the task from the structure of your prompt.
# - In zero-shot prompting, you only provide the structure to the model, but without any examples of the completed task.
# 

# In[ ]:


prompt = """
Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: ?
"""
response = llama(prompt)
print(response)

# ### Few-shot Prompting
# - Here is an example of few-shot prompting.
# - In few-shot prompting, you not only provide the structure to the model, but also two or more examples.
# - You are prompting the model to see if it can infer the task from the structure, as well as the examples in your prompt.

# In[ ]:


prompt = """
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative

Message: Can't wait to order pizza for dinner tonight
Sentiment: Positive

Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: ?
"""
response = llama(prompt)
print(response)

# ### Specifying the Output Format
# - You can also specify the format in which you want the model to respond.
# - In the example below, you are asking to "give a one word response".

# In[ ]:


prompt = """
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative

Message: Can't wait to order pizza for dinner tonight
Sentiment: Positive

Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: ?

Give a one word response.
"""
response = llama(prompt)
print(response)

# **Note:** For all the examples above, you used the 7 billion parameter model, `llama-2-7b-chat`. And as you saw in the last example, the 7B model was uncertain about the sentiment.
# 
# - You can use the larger (70 billion parameter) `llama-2-70b-chat` model to see if you get a better, certain response:

# In[ ]:


prompt = """
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative

Message: Can't wait to order pizza for dinner tonight
Sentiment: Positive

Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: ?

Give a one word response.
"""
response = llama(prompt,
                model="togethercomputer/llama-2-70b-chat")
print(response)

# - Now, use the smaller model again, but adjust your prompt in order to help the model to understand what is being expected from it.
# - Restrict the model's output format to choose from `positive`, `negative` or `neutral`.

# In[ ]:


prompt = """
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative

Message: Can't wait to order pizza for dinner tonight
Sentiment: Positive

Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: 

Respond with either positive, negative, or neutral.
"""
response = llama(prompt)
print(response)

# ### Role Prompting
# - Roles give context to LLMs what type of answers are desired.
# - Llama 2 often gives more consistent responses when provided with a role.
# - First, try standard prompt and see the response.

# In[ ]:


prompt = """
How can I answer this question from my friend:
What is the meaning of life?
"""
response = llama(prompt)
print(response)

# - Now, try it by giving the model a "role", and within the role, a "tone" using which it should respond with.

# In[ ]:


role = """
Your role is a life coach \
who gives advice to people about living a good life.\
You attempt to provide unbiased advice.
You respond in the tone of an English pirate.
"""

prompt = f"""
{role}
How can I answer this question from my friend:
What is the meaning of life?
"""
response = llama(prompt)
print(response)

# ### Summarization
# - Summarizing a large text is another common use case for LLMs. Let's try that!

# In[ ]:


email = """
Dear Amit,

An increasing variety of large language models (LLMs) are open source, or close to it. The proliferation of models with relatively permissive licenses gives developers more options for building applications.

Here are some different ways to build applications based on LLMs, in increasing order of cost/complexity:

Prompting. Giving a pretrained LLM instructions lets you build a prototype in minutes or hours without a training set. Earlier this year, I saw a lot of people start experimenting with prompting, and that momentum continues unabated. Several of our short courses teach best practices for this approach.
One-shot or few-shot prompting. In addition to a prompt, giving the LLM a handful of examples of how to carry out a task — the input and the desired output — sometimes yields better results.
Fine-tuning. An LLM that has been pretrained on a lot of text can be fine-tuned to your task by training it further on a small dataset of your own. The tools for fine-tuning are maturing, making it accessible to more developers.
Pretraining. Pretraining your own LLM from scratch takes a lot of resources, so very few teams do it. In addition to general-purpose models pretrained on diverse topics, this approach has led to specialized models like BloombergGPT, which knows about finance, and Med-PaLM 2, which is focused on medicine.
For most teams, I recommend starting with prompting, since that allows you to get an application working quickly. If you’re unsatisfied with the quality of the output, ease into the more complex techniques gradually. Start one-shot or few-shot prompting with a handful of examples. If that doesn’t work well enough, perhaps use RAG (retrieval augmented generation) to further improve prompts with key information the LLM needs to generate high-quality outputs. If that still doesn’t deliver the performance you want, then try fine-tuning — but this represents a significantly greater level of complexity and may require hundreds or thousands more examples. To gain an in-depth understanding of these options, I highly recommend the course Generative AI with Large Language Models, created by AWS and DeepLearning.AI.

(Fun fact: A member of the DeepLearning.AI team has been trying to fine-tune Llama-2-7B to sound like me. I wonder if my job is at risk? 😜)

Additional complexity arises if you want to move to fine-tuning after prompting a proprietary model, such as GPT-4, that’s not available for fine-tuning. Is fine-tuning a much smaller model likely to yield superior results than prompting a larger, more capable model? The answer often depends on your application. If your goal is to change the style of an LLM’s output, then fine-tuning a smaller model can work well. However, if your application has been prompting GPT-4 to perform complex reasoning — in which GPT-4 surpasses current open models — it can be difficult to fine-tune a smaller model to deliver superior results.

Beyond choosing a development approach, it’s also necessary to choose a specific model. Smaller models require less processing power and work well for many applications, but larger models tend to have more knowledge about the world and better reasoning ability. I’ll talk about how to make this choice in a future letter.

Keep learning!

Andrew
"""

# In[ ]:


prompt = f"""
Summarize this email and extract some key points.
What did the author say about llama models?:

email: {email}
"""

response = llama(prompt)
print(response)

# ### Providing New Information in the Prompt
# - A model's knowledge of the world ends at the moment of its training - so it won't know about more recent events.
# - Llama 2 was released for research and commercial use on July 18, 2023, and its training ended some time before that date.
# - Ask the model about an event, in this case, FIFA Women's World Cup 2023, which started on July 20, 2023, and see how the model responses.

# In[ ]:


prompt = """
Who won the 2023 Women's World Cup?
"""
response = llama(prompt)
print(response)

# - As you can see, the model still thinks that the tournament is yet to be played, even though you are now in 2024!
# - Another thing to **note** is, July 18, 2023 was the date the model was released to public, and it was trained even before that, so it only has information upto that point. The response says, "the final match is scheduled to take place in July 2023", but the final match was played on August 20, 2023.

# - You can provide the model with information about recent events, in this case text from Wikipedia about the 2023 Women's World Cup.

# In[ ]:


context = """
The 2023 FIFA Women's World Cup (Māori: Ipu Wahine o te Ao FIFA i 2023)[1] was the ninth edition of the FIFA Women's World Cup, the quadrennial international women's football championship contested by women's national teams and organised by FIFA. The tournament, which took place from 20 July to 20 August 2023, was jointly hosted by Australia and New Zealand.[2][3][4] It was the first FIFA Women's World Cup with more than one host nation, as well as the first World Cup to be held across multiple confederations, as Australia is in the Asian confederation, while New Zealand is in the Oceanian confederation. It was also the first Women's World Cup to be held in the Southern Hemisphere.[5]
This tournament was the first to feature an expanded format of 32 teams from the previous 24, replicating the format used for the men's World Cup from 1998 to 2022.[2] The opening match was won by co-host New Zealand, beating Norway at Eden Park in Auckland on 20 July 2023 and achieving their first Women's World Cup victory.[6]
Spain were crowned champions after defeating reigning European champions England 1–0 in the final. It was the first time a European nation had won the Women's World Cup since 2007 and Spain's first title, although their victory was marred by the Rubiales affair.[7][8][9] Spain became the second nation to win both the women's and men's World Cup since Germany in the 2003 edition.[10] In addition, they became the first nation to concurrently hold the FIFA women's U-17, U-20, and senior World Cups.[11] Sweden would claim their fourth bronze medal at the Women's World Cup while co-host Australia achieved their best placing yet, finishing fourth.[12] Japanese player Hinata Miyazawa won the Golden Boot scoring five goals throughout the tournament. Spanish player Aitana Bonmatí was voted the tournament's best player, winning the Golden Ball, whilst Bonmatí's teammate Salma Paralluelo was awarded the Young Player Award. England goalkeeper Mary Earps won the Golden Glove, awarded to the best-performing goalkeeper of the tournament.
Of the eight teams making their first appearance, Morocco were the only one to advance to the round of 16 (where they lost to France; coincidentally, the result of this fixture was similar to the men's World Cup in Qatar, where France defeated Morocco in the semi-final). The United States were the two-time defending champions,[13] but were eliminated in the round of 16 by Sweden, the first time the team had not made the semi-finals at the tournament, and the first time the defending champions failed to progress to the quarter-finals.[14]
Australia's team, nicknamed the Matildas, performed better than expected, and the event saw many Australians unite to support them.[15][16][17] The Matildas, who beat France to make the semi-finals for the first time, saw record numbers of fans watching their games, their 3–1 loss to England becoming the most watched television broadcast in Australian history, with an average viewership of 7.13 million and a peak viewership of 11.15 million viewers.[18]
It was the most attended edition of the competition ever held.
"""

# In[ ]:


prompt = f"""
Given the following context, who won the 2023 Women's World cup?
context: {context}
"""
response = llama(prompt)
print(response)

# ### Try it Yourself!
# 
# Try asking questions of your own! Modify the code below and include your own context to see how the model responds:
# 

# In[ ]:


context = """
<paste context in here>
"""
query = "<your query here>"

prompt = f"""
Given the following context,
{query}

context: {context}
"""
response = llama(prompt,
                 verbose=True)
print(response)

# ### Chain-of-thought Prompting
# - LLMs can perform better at reasoning and logic problems if you ask them to break the problem down into smaller steps. This is known as **chain-of-thought** prompting.

# In[ ]:


prompt = """
15 of us want to go to a restaurant.
Two of them have cars
Each car can seat 5 people.
Two of us have motorcycles.
Each motorcycle can fit 2 people.

Can we all get to the restaurant by car or motorcycle?
"""
response = llama(prompt)
print(response)

# - Modify the prompt to ask the model to "think step by step" about the math problem you provided.

# In[ ]:


prompt = """
15 of us want to go to a restaurant.
Two of them have cars
Each car can seat 5 people.
Two of us have motorcycles.
Each motorcycle can fit 2 people.

Can we all get to the restaurant by car or motorcycle?

Think step by step.
"""
response = llama(prompt)
print(response)

# - Provide the model with additional instructions.

# In[ ]:


prompt = """
15 of us want to go to a restaurant.
Two of them have cars
Each car can seat 5 people.
Two of us have motorcycles.
Each motorcycle can fit 2 people.

Can we all get to the restaurant by car or motorcycle?

Think step by step.
Explain each intermediate step.
Only when you are done with all your steps,
provide the answer based on your intermediate steps.
"""
response = llama(prompt)
print(response)

# - The order of instructions matters!
# - Ask the model to "answer first" and "explain later" to see how the output changes.

# In[ ]:


prompt = """
15 of us want to go to a restaurant.
Two of them have cars
Each car can seat 5 people.
Two of us have motorcycles.
Each motorcycle can fit 2 people.

Can we all get to the restaurant by car or motorcycle?
Think step by step.
Provide the answer as a single yes/no answer first.
Then explain each intermediate step.
"""

response = llama(prompt)
print(response)

# - Since LLMs predict their answer one token at a time, the best practice is to ask them to think step by step, and then only provide the answer after they have explained their reasoning.
