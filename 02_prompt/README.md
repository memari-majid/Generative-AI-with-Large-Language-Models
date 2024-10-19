
# Chapter 2: Prompt Engineering and In-Context Learning

This chapter focuses on **prompt engineering** and **in-context learning** for generative AI models, which are essential for effectively interacting with models like LLaMA, Code Llama, and FLAN-T5. These techniques help guide models to produce better outputs and perform tasks more accurately by crafting appropriate prompts and providing contextual examples.

## Key Concepts Covered in This Chapter

**Prompt Engineering:** This involves designing input prompts that influence the model to generate high-quality responses. Prompts typically include components like instructions, context, and examples (in one-shot or few-shot form).

**In-Context Learning:** A technique where examples of tasks (prompt-completion pairs) are provided to the model within the input prompt, allowing the model to temporarily learn task-specific behavior without changing its parameters. This technique is used in both one-shot and few-shot scenarios.

**Chain-of-Thought (CoT) Prompting:** An advanced technique where prompts are crafted to encourage the model to break down complex tasks step-by-step, leading to more accurate results for multi-step problems.

**Generative Parameters:** The model's response quality and randomness can be controlled by adjusting parameters such as temperature, top-k, and top-p, which are important for balancing coherence and creativity.

## Code Example 1: LLaMA 3.2 for Text Generation

This example demonstrates how to use the LLaMA 3.2-1B model for text generation by setting up the model and tokenizer from Hugging Face. It showcases the importance of generation configurations like `max_new_tokens`, `temperature`, and `top_p` in controlling the quality and diversity of the model's output.

```python
# Example of a prompt:
messages = "[INST] How will AI impact the future of healthcare? [/INST]"

# Call the model to generate a response:
response = generate_response(messages)
```

## Code Example 2: Chain-of-Thought (CoT) Prompting

This example builds on the idea of **Chain-of-Thought** prompting by alternating between user and assistant messages in a multi-turn conversation. It explains how to structure input prompts using conversation markers and highlights the impact of reasoning prompts (step-by-step explanations) on improving the model's task-solving abilities.

```python
# Prompt for CoT reasoning:
prompt = "[INST] 15 of us want to go to a restaurant. Two of them have cars. Can we all get there? Think step by step. [/INST]"

# The model processes the reasoning and provides a detailed answer.
```

## Code Example 3: Prompt Engineering Techniques

This example covers various prompt engineering techniques such as **zero-shot**, **few-shot**, and task-specific prompting. It illustrates how providing examples in the prompt helps the model infer the task and perform better. Chain-of-Thought (CoT) techniques are also demonstrated to encourage the model to solve problems step-by-step, ensuring clarity and structured output.

```python
# Example of few-shot prompting:
prompt = '''
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative

Message: Can't wait to order pizza for dinner tonight
Sentiment: Positive

Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: ?
'''
```

## Code Example 4: Code Llama for Code Generation

This example uses **Code Llama** to generate Python code. The script demonstrates asking the model to solve specific programming tasks, such as finding the minimum and maximum values in a list of temperatures or generating recursive and non-recursive functions.

```python
# Example prompt for generating a Python function:
prompt = "Write Python code that calculates the minimum of the list temp_min and the maximum of the list temp_max."
```

## Code Example 5: Llama Guard for Safety

The **Llama Guard** model is introduced to ensure that user inputs and model outputs adhere to safety policies. This example highlights checking conversations for unsafe content (violence, hate speech, criminal planning, etc.) before processing them further.

```python
# Example unsafe query:
query = "I'm so unhappy with this pizza that I want to hit the chef on the head with a breadstick!"

# Llama Guard checks if this query violates safety policies.
```

## Code Example 6: FLAN-T5 for Summarization

This example uses the **FLAN-T5 model** for dialogue summarization, demonstrating zero-shot and few-shot inference. It highlights the impact of providing examples to the model and improving summarization through prompt engineering.

```python
# Example dialogue for summarization:
prompt = "Dialogue: Hello! How are you? I wanted to check in about the meeting. Can we reschedule? What was going on?"
```

By experimenting with these examples, the chapter teaches how to effectively use generative AI models through prompt engineering, in-context learning, and safety mechanisms.
