#!/usr/bin/env python
# coding: utf-8

# # Llama2 Chatbot in Hugging Face and LangChain
# 
# In this notebook we'll explore how we can use the open source **Llama-70b-chat** model in both Hugging Face transformers and LangChain.
# At the time of writing, you must first request access to Llama 2 models via [this form]() (access is typically granted within a few hours).
# 
# We start by doing a `pip install` of all required libraries.

# In[2]:


# !pip install -qU \
#     transformers==4.31.0 \
#     accelerate==0.21.0 \
#     einops==0.6.1 \
#     langchain==0.0.309 \
#     xformers==0.0.20 \
#     bitsandbytes==0.41.0 \
#     torch==2.0.1

# ## Initializing the Hugging Face Pipeline

# The first thing we need to do is initialize a `text-generation` pipeline with Hugging Face transformers. The Pipeline requires three things that we must initialize first, those are:
# 
# * A LLM, in this case it will be a variant of `meta-llama/Llama-2-70b-chat-hf`.
# 
# * The respective tokenizer for the model.
# 
# We'll explain these as we get to them, let's begin with our model.
# 
# We initialize the model and move it to our CUDA-enabled GPU. Using Colab this can take 5-10 minutes to download and initialize the model.

# In[3]:


from torch import cuda, bfloat16
import transformers

model_id = "NousResearch/Llama-2-7b-chat-hf" # not gated

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
#hf_auth = '<YOUR_API_KEY>'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
#    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
#    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

# The pipeline requires a tokenizer which handles the translation of human readable plaintext to LLM readable token IDs. The Llama 2 70B models were trained using the Llama 2 70B tokenizer, which we initialize like so:

# In[4]:


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
#    use_auth_token=hf_auth
)

# Now we're ready to initialize the HF pipeline. There are a few additional parameters that we must define here. Comments explaining these have been included in the code.

# In[5]:


generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

# Confirm this is working:

# In[6]:


res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])

# Now to implement this in LangChain

# In[7]:


from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)

# In[8]:


llm(prompt="Explain to me the difference between nuclear fission and fusion.")

# We still get the same output as we're not really doing anything differently here, but we have now added **Llama 2 70B Chat** to the LangChain library. Using this we can now begin using LangChain's advanced agent tooling, chains, etc, with **Llama 2**.

# ## Initializing an Conversational Agent
# 
# Getting a conversational agent to work with open source models is incredibly hard. However, with Llama 2 70B it is now possible. Let's see how we can get it running!
# 
# We first need to initialize the agent. Conversational agents require several things such as conversational `memory`, access to `tools`, and an `llm` (which we have already initialized).

# In[9]:


from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
)
tools = load_tools(["llm-math"], llm=llm)

# In[10]:


from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish

class OutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            # this will work IF the text is a valid JSON with action and action_input
            response = parse_json_markdown(text)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                # this means the agent is finished so we call AgentFinish
                return AgentFinish({"output": action_input}, text)
            else:
                # otherwise the agent wants to use an action, so we call AgentAction
                return AgentAction(action, action_input, text)
        except Exception:
            # sometimes the agent will return a string that is not a valid JSON
            # often this happens when the agent is finished
            # so we just return the text as the output
            return AgentFinish({"output": text}, text)

    @property
    def _type(self) -> str:
        return "conversational_chat"

# initialize output parser for agent
parser = OutputParser()

# In[11]:


from langchain.agents import initialize_agent

# initialize agent
agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    early_stopping_method="generate",
    memory=memory,
    #agent_kwargs={"output_parser": parser}
)

# In[12]:


agent.agent.llm_chain.prompt

# We need to add special tokens used to signify the beginning and end of instructions, and the beginning and end of system messages. These are described in the Llama-2 model cards on Hugging Face.

# In[13]:


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# In[14]:


sys_msg = B_SYS + """Assistant is a expert JSON builder designed to assist with a wide range of tasks.

Assistant is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters.

All of Assistant's communication is performed using this JSON format.

Assistant can also use tools by responding to the user with tool use instructions in the same "action" and "action_input" JSON format. Tools available to Assistant are:

- "Calculator": Useful for when you need to answer questions about math.
  - To use the calculator tool, Assistant should write like so:
    ```json
    {{"action": "Calculator",
      "action_input": "sqrt(4)"}}
    ```

Here are some previous conversations between the Assistant and User:

User: Hey how are you today?
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "I'm good thanks, how are you?"}}
```
User: I'm great, what is the square root of 4?
Assistant: ```json
{{"action": "Calculator",
 "action_input": "sqrt(4)"}}
```
User: 2.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "It looks like the answer is 2!"}}
```
User: Thanks could you tell me what 4 to the power of 2 is?
Assistant: ```json
{{"action": "Calculator",
 "action_input": "4**2"}}
```
User: 16.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "It looks like the answer is 16!"}}
```

Here is the latest conversation between Assistant and User.""" + E_SYS
new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
agent.agent.llm_chain.prompt = new_prompt

# In the Llama 2 paper they mentioned that it was difficult to keep Llama 2 chat models following instructions over multiple interactions. One way they found that improves this is by inserting a reminder of the instructions to each user query. We do that here:

# In[15]:


instruction = B_INST + " Respond to the following in JSON with 'action' and 'action_input' values " + E_INST
human_msg = instruction + "\nUser: {input}"

agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg

# In[16]:


agent.agent.llm_chain.prompt

# Now we can begin asking questions...

# In[17]:


agent("hey how are you today?")

# In[ ]:


agent("what is 4 to the power of 2.1?")

# In[ ]:


agent("can you multiply that by 3?")

# With that we have our **open source** conversational agent running on Colab with ~38GB of RAM.
# 
# ---
