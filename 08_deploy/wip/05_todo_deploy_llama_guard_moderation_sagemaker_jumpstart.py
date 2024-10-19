#!/usr/bin/env python
# coding: utf-8

# # Introduction to SageMaker JumpStart - text moderation with Llama Guard

# ## Setup
# ***

# In[7]:


%pip install -U sagemaker

# ## Deploy model
# 
# ***
# You can now deploy the model using SageMaker JumpStart. The following code uses the default instance `ml.g5.2xlarge` for the inference endpoint You can deploy the model on other instance types by passing `instance_type` in the `JumpStartModel` class. For successful deployment, you must manually change the `accept_eula` argument in the model's deploy method to `True`. This model is deployed using the text-generation-inference (TGI) deep learning container. The deployment might take few minutes. 
# ***

# In[10]:


model_id = "meta-textgeneration-llama-guard-7b"
model_version = "1.*"

# In[11]:


from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(model_id=model_id, model_version=model_version)

# In[12]:


try:
    predictor = model.deploy(accept_eula=True)
except Exception as e:
    print(e)

# ## Invoke the endpoint
# ***
# 
# ### Supported parameters
# 
# ***
# This model supports many parameters while performing inference. They include:
# 
# * **max_length:** Model generates text until the output length (which includes the input context length) reaches `max_length`. If specified, it must be a positive integer.
# * **max_new_tokens:** Model generates text until the output length (excluding the input context length) reaches `max_new_tokens`. If specified, it must be a positive integer.
# * **num_beams:** Number of beams used in the greedy search. If specified, it must be integer greater than or equal to `num_return_sequences`.
# * **no_repeat_ngram_size:** Model ensures that a sequence of words of `no_repeat_ngram_size` is not repeated in the output sequence. If specified, it must be a positive integer greater than 1.
# * **temperature:** Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If `temperature` -> 0, it results in greedy decoding. If specified, it must be a positive float.
# * **early_stopping:** If True, text generation is finished when all beam hypotheses reach the end of sentence token. If specified, it must be boolean.
# * **do_sample:** If True, sample the next word as per the likelihood. If specified, it must be boolean.
# * **top_k:** In each step of text generation, sample from only the `top_k` most likely words. If specified, it must be a positive integer.
# * **top_p:** In each step of text generation, sample from the smallest possible set of words with cumulative probability `top_p`. If specified, it must be a float between 0 and 1.
# * **return_full_text:** If True, input text will be part of the output generated text. If specified, it must be boolean. The default value for it is False.
# * **stop**: If specified, it must a list of strings. Text generation stops if any one of the specified strings is generated.
# 
# We may specify any subset of the parameters mentioned above while invoking an endpoint. Next, we show an example of how to invoke endpoint with these arguments
# ***

# ### Example prompts
# ***
# The examples in this section demonstrate how to perform text generation with conversational dialog as prompt inputs. Example payloads are retrieved programmatically from the `JumpStartModel` object.
# 
# Similar to Llama-2, Llama Guard uses special tokens to indicate safety instructions to the model. In general, the payload should follow the format `<s>[INST] {user_prompt} [/INST]`. This user prompt can further include sections for content category definitions and conversations, which looks like the following:
# 
# ```
# <s>[INST] {task}
# 
# <BEGIN UNSAFE CONTENT CATEGORIES>
# {unsafe_content_categories}
# <END UNSAFE CONTENT CATEGORIES>
# 
# <BEGIN CONVERSATION>
# {conversation}
# <END CONVERSATION}
# 
# {instruction} [/INST]
# ```
# 
# In the next section we discuss the recommended default values for the task, content category, and instruction definitions. The conversation should alternate between `User` and `Agent` text as follows:
# 
# ```
# User: {user_input_1}
# 
# Agent: {optional_agent_response_1}
# 
# User: {optional_user_input_2}
# 
# ...
# ```
# 
# 
# ***
# 

# In[ ]:


example_payloads = model.retrieve_all_examples()

for payload in example_payloads:
    try:
        response = predictor.predict(payload.body)
        print(
            "\nInput\n",
            payload.body,
            "\n\nOutput\n",
            response[0]["generated_text"],
            "\n\n===============",
        )
    except Exception as e:
        print(e)

# ## Moderate a conversation with Llama-2 Chat
# ***
# You can now deploy a Llama-2 7B Chat model endpoint for conversational chat and then use Llama Guard to moderate input and output text coming from Llama-2 7B Chat. 
# 
# We show you the example of Llama-2 7B chat model’s input and output moderated through Llama Guard, but you may use Llama Guard for moderation with any LLM of your choice.
# ***

# In[ ]:


llm_model_id = "meta-textgeneration-llama-2-7b-f"
llm_model_version = "3.*"

# In[ ]:


model_llm = JumpStartModel(model_id=llm_model_id, model_version=llm_model_version)

# In[ ]:


try:
    predictor_llm = model_llm.deploy(accept_eula=True)
except Exception as e:
    print(e)

# ***
# You can now define the Llama Guard task template. The unsafe content categories may be adjusted as desired for your specific use case. You can define in plain text the meaning of each content category, to include which content should be flagged as unsafe and which content should be permitted as safe.
# ***

# In[ ]:


TASK = "Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy with the below categories."

INSTRUCTION = """\
Provide your safety assessment for Agent in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories."""

UNSAFE_CONTENT_CATEGORIES = """\
O1: Violence and Hate. 
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
- Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence.
O2: Sexual Content.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
O3: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
O4: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Discuss firearms and the arguments for and against firearm ownership.
O5: Regulated or Controlled Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
O6: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves."""

# ***
# Next, we define helper functions `format_chat_messages` and `format_guard_messages` to format the prompt for chat model and for Llama Guard model that required special tokens. 
# ***

# In[ ]:


from itertools import cycle
from typing import Dict, List


def format_chat_messages(messages: List[Dict[str, str]]) -> List[str]:
    """Format messages for Llama-2 chat models.

    The model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and
    alternating (u/a/u/a/u...). The last message must be from 'user'.
    """
    prompt: List[str] = []

    if messages[0]["role"] == "system":
        content = "".join(
            ["<<SYS>>\n", messages[0]["content"], "\n<</SYS>>\n\n", messages[1]["content"]]
        )
        messages = [{"role": messages[1]["role"], "content": content}] + messages[2:]

    for user, answer in zip(messages[::2], messages[1::2]):
        prompt.extend(
            [
                "<s>",
                "[INST] ",
                (user["content"]).strip(),
                " [/INST] ",
                (answer["content"]).strip(),
                "</s>",
            ]
        )

    prompt.extend(["<s>", "[INST] ", (messages[-1]["content"]).strip(), " [/INST] "])

    return "".join(prompt)


def format_guard_messages(
    messages: List[Dict[str, str]],
    task: str = TASK,
    instruction: str = INSTRUCTION,
    unsafe_content_categories: str = UNSAFE_CONTENT_CATEGORIES,
) -> List[str]:
    """Format messages for Llama Guard models.

    The model only supports 'user' and 'assistant' roles, and must alternate (u/a/u/a/u...).
    """
    turn = cycle(["User", "Agent"])

    return "".join(
        [
            "[INST] ",
            task,
            "\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\n",
            unsafe_content_categories,
            "\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>",
            *[f"\n\n{next(turn)}: {message['content']}" for message in messages],
            "\n\n<END CONVERSATION>\n\n",
            instruction,
            " [/INST]",
        ]
    )

# ***
# You can then use these helper functions on an example message input prompt to run the example input through Llama Guard to determine if the message content is safe.
# ***

# In[ ]:


messages_input = [
    {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"}
]
payload_input_guard = {"inputs": format_guard_messages(messages_input)}

try:
    response_input_guard = predictor.predict(payload_input_guard)
    assert response_input_guard[0]["generated_text"].strip() == "safe"
    print(response_input_guard)
except Exception as e:
    print(e)

# ***
# We see below output that indicates that message is safe. You may notice that the prompt includes words that may be associated with violence, but, in this case, Llama Guard is able to understand the context with respect to the instructions and unsafe category definitions we provided above that it's a safe prompt and not related to violence.  
# 
# Now that you have asserted the input text is determined to be safe with respect to your Llama Guard content categories, you can pass this payload to the deployed Llama-2 7B model to generate text.
# ***

# In[ ]:


payload_input_llm = {
    "inputs": format_chat_messages(messages_input),
    "parameters": {"max_new_tokens": 128},
}

try:
    response_llm = predictor_llm.predict(payload_input_llm)
    print(response_llm)
except Exception as e:
    print(e)

# ***
# Finally, you may wish to confirm that the response text from the model is determined to contain safe content. Here, you extend the LLM output response to the input messages and run this whole conversation through Llama Guard to ensure the conversation is safe for your application.
# ***

# In[ ]:


try:
    messages_output = messages_input.copy()
    messages_output.extend([{"role": "assistant", "content": response_llm[0]["generated_text"]}])
    payload_output = {"inputs": format_guard_messages(messages_output)}

    response_output_guard = predictor.predict(payload_output)

    assert response_output_guard[0]["generated_text"].strip() == "safe"
    print(response_output_guard)
except Exception as e:
    print(e)

# ## Clean up the endpoint

# In[ ]:


try:
    predictor.delete_model()
    predictor.delete_endpoint()
    predictor_llm.delete_model()
    predictor_llm.delete_endpoint()
except Exception as e:
    print(e)

# ## Notebook CI Test Results
# 
# This notebook was tested in multiple regions. The test results are as follows, except for us-west-2 which is shown at the top of the notebook.
# 
# 
# ![This us-east-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-east-1/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This us-east-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-east-2/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This us-west-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-west-1/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This ca-central-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ca-central-1/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This sa-east-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/sa-east-1/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This eu-west-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-1/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This eu-west-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-2/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This eu-west-3 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-3/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This eu-central-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-central-1/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This eu-north-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-north-1/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This ap-southeast-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-southeast-1/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This ap-southeast-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-southeast-2/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This ap-northeast-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-northeast-1/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This ap-northeast-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-northeast-2/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# ![This ap-south-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-south-1/introduction_to_amazon_algorithms|jumpstart-foundation-models|llama-guard-text-moderation.ipynb)
# 
# 
