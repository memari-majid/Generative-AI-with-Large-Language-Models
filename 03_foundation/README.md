Chapter 2: Prompt Engineering and In-Context Learning - Detailed Summary
In this chapter, you are introduced to low-code methods for interacting with generative AI models, specifically through prompt engineering and in-context learning. These techniques enable users to optimize model responses by crafting prompts and providing relevant context. Writing effective prompts is described as both an art and a science, requiring iteration to guide the model toward producing more accurate and applicable responses.

Prompt Engineering
Generative AI models accept prompts as input, which typically include instructions or context that define a task. For instance, a prompt could instruct the model to “summarize the following text” or answer factual questions like “Who won the 2016 baseball World Series?”. The model generates a completion based on the prompt, which can vary in content type, such as text, image, video, or audio. The goal of prompt engineering is to create prompts that elicit the most useful completions from the model.

A key point in the chapter is that prompts are not processed as raw text. Instead, they are tokenized into fragments called tokens, which are the basic units that the model uses for computations. Tokenization allows the model to handle sequences of text in a compact form, with vocabularies often consisting of thousands of tokens. This token-based processing is fundamental to how models interpret and generate language.

In-Context Learning
Another major concept covered in this chapter is in-context learning. This technique involves passing multiple prompt-completion pairs along with your input prompt, thereby providing examples for the model to learn from during that specific interaction. By offering these pairs as context, the model is nudged to generate responses that follow the patterns of the examples provided. This is a powerful feature of generative models, as it allows them to temporarily adjust their behavior without any changes to their underlying parameters.

In-context learning can be done in zero-shot, one-shot, or few-shot modes, depending on the number of example pairs provided. In zero-shot mode, no examples are given, and the model relies on its pre-existing knowledge to answer a prompt. In one-shot and few-shot modes, one or more example prompt-completion pairs are provided, helping the model generate responses more aligned with the examples.

Configurable Parameters
The chapter also introduces common configurable parameters that control the creativity of a model’s response. These parameters allow for fine-tuning of the model's output:

Temperature: Controls the randomness of the response. A higher temperature results in more creative and diverse responses, while a lower temperature generates more focused and deterministic completions.
Top-k: Limits the model’s choice to the top k tokens with the highest probabilities. A smaller k value makes the model less creative by restricting its options to fewer tokens.
Top-p: Also known as nucleus sampling, this parameter restricts token selection to those whose cumulative probability adds up to a value p. It allows for more variability in responses without selecting improbable tokens.
Best Practices in Prompt Engineering
The chapter concludes by sharing several best practices for crafting effective prompts:

Be clear and concise: Avoid ambiguity in your prompts, as unclear instructions may confuse the model and result in suboptimal responses.
Provide context: Including relevant information helps the model understand the task or topic better, leading to more accurate responses.
Explicit instructions: Specify any desired output format clearly in your prompt to ensure that the model generates the correct type of response.
Avoid negative formulations: Use straightforward, positive language when defining tasks, as this minimizes confusion in the model’s interpretation.
Use few-shot examples: When appropriate, include example prompt-completion pairs to guide the model’s behavior. This technique improves the coherence and relevance of the generated responses.
Step-by-step reasoning: For complex tasks, ask the model to “think step-by-step” to ensure it follows a logical process when generating a response.
Overall, prompt engineering is a learned skill that requires experimentation, and in-context learning provides a useful mechanism for guiding model behavior. With practice, you can master these techniques to better interact with generative AI models, making them more effective in real-world tasks.
