# Chapter 2. Prompt Engineering and In-Context Learning

In this chapter, you will learn about low-code ways to interact with generative AI models—specifically, prompt engineering and in-context learning. You will see that writing prompts is both an art and a science that helps the model generate better and more applicable responses. We also provide some best practices when defining prompts and prompt templates to get the most out of your generative models.

You will also learn how to use in-context learning to pass multiple prompt-completion pairs (e.g., question-answer pairs) in the “context” along with your prompt input. This in-context learning nudges the model to respond similarly to the prompt-completion pairs in the context. This is one of the more remarkable capabilities of generative models as it temporarily alters the model’s behavior for the duration of just that single request.

Lastly, you will learn some of the most commonly configured generative parameters like temperature and top-k that control the generative model’s creativity when creating content.

## Prompts and Completions

Language-based generative models accept prompts as input and generate a completion. These prompts and completions are made up of text-based tokens, as you will see next.

While generative AI tasks can span multiple content modalities, they often involve a text-based input. This input is called a prompt and includes the instructions, context, and any constraints used to accomplish a given task.

Some examples of prompt instructions are “Summarize the following text” or “Who won the baseball World Series in 2016?” The model then responds with a “completion” that returns the result of the task. This completion is often text-based, but it could be any content type the model is trained to output, such as text, image, video, or audio. You will learn how to optimize your prompts to achieve your desired completions later in this chapter on text-based prompts as well as in the coverage of multimodal prompts in Chapter 11.

## Tokens

It’s important to note that while text-based prompts and completions are implemented and interpreted by humans as natural language sentences, generative models convert them into sequences of tokens, or word fragments. By combining many of these tokens in different ways, the model is capable of representing an exponential number of words using a relatively small number of tokens—often on the order of 30,000–100,000 tokens in the model’s vocabulary.

**Tip:** As a rule of thumb, it’s common to approximate 1.3 tokens per word, but this multiplier can vary. You would use the 1.3 multiplier when estimating the cost of services that use token-based pricing. Often, these services charge per million tokens.

Using this small vocabulary, a language model is capable of learning and understanding human language during the model pretraining phase. During pretraining, the model sees millions of documents as part of the training dataset. From the model’s standpoint, a document is simply just a sequence of tokens from the model’s vocabulary. You will learn more about the model pretraining phase in Chapter 3.

Machine learning and AI models, including generative AI models, rely on statistics and linear algebra for their computations, including probability modeling, loss functions, and matrix multiplications. These calculations power all deep learning operations, and they prefer to work with numbers and not raw text, images, or videos.

## Prompt Engineering

Prompt engineering is a new and exciting skill focused on how to better understand and apply generative models to your tasks and use cases. Effective prompt engineering helps you get the most out of your generative AI models and applications.

The input that you send into a generative model is typically called the prompt. The prompt can include text for large language models, or other modalities, such as images, video, for multimodal models. This prompt is passed to the model during inference time to generate a “completion.”

Here is a simple example question-answer prompt and completion.

**Prompt:**

Who won the 2016 baseball World Series?

**Completion:**

The Chicago Cubs won the 2016 baseball World Series, defeating the Cleveland Indians in a thrilling seven-game series. It was the Cubs' first World Series title in 108 years, ending the longest championship drought in Major League Baseball history.

You may have to experiment with your prompt several times to get a proper and precise response, as some of these generative models are quite chatty. Prompt engineering is a learned skill that requires many iterations across many different model types and linguistic nuances. These nuances often depend on how the model was trained.

Most modern human-facing chat models have been fine-tuned using some form of human-labeled data—often with reinforcement learning, which you will explore in Chapter 7. Chat models sometimes include extra markup in the prompt and completion to indicate the role and distinguish between the user input (for example, with User:) and the model response (for example, with Assistant:). In prompt terminology, User: is often referred to as the input indicator and Assistant: as the output indicator.

Here is the same example question-answer prompt and completion, this time using a model that has been trained to act as an AI assistant and uses input and output indicators.

**Prompt:**

User: Who won the 2016 baseball World Series?

**Completion:**

Assistant: The Chicago Cubs won the 2016 baseball World Series, defeating the Cleveland Indians in a thrilling seven-game series. It was the Cubs' first World Series title in 108 years, ending the longest championship drought in Major League Baseball history.

Note that prompt structures are very model-specific. Using different input and output indicators may result in “off-distribution” and undesirable results. Therefore, always look up the prompt structure when you start experimenting with a new generative AI model. You can usually find this information in model documentation such as the model card.

Let’s explore some common prompt structures and techniques to get the most out of off-the-shelf generative AI models.

## Prompt Structure

The prompt structure used in the previous example is a simple chat-assistant structure. A more complete prompt structure usually includes a section for each of the following: instruction, context, input data, and output indicator. Let’s discuss the two additional prompt elements, instruction and context.

### Instruction

An instruction is a piece of text that you pass to the model that describes the task you want the model to perform. For example, you can add an instruction to our previous example to guide the model to generate a one-sentence summary.

**Prompt:**

User: Answer the following question in one sentence.  
Who won the 2016 baseball World Series?

**Completion:**

Assistant: The Chicago Cubs won the 2016 baseball World Series, defeating the Cleveland Indians in a thrilling seven-game series that ended with a final score of 8-7 in extra innings.

And indeed, the model responded with a single sentence.

Writing effective instructions for generative AI models involves clarity, specificity, and context to guide the model toward the desired output. Instructions should be straightforward and easy to understand. The more specific your instruction, the better the model can fulfill your request. Providing relevant context can help the model better understand the task or topic at hand.

### Context

Context refers to relevant information or details that you pass to the model, so it better understands the task or topic and responds appropriately. This context can include preceding dialogue, background information, specific user requests, or any data that provides a frame of reference for the model to generate more contextually relevant and accurate responses. Passing context allows for more coherent and meaningful interactions with the model.

A popular technique to guide the model’s response toward desired output is to share example prompt-completion pairs as context information. Depending on how many examples you provide, this is called one-shot or few-shot inference. The model’s ability to learn from those examples and adapt its responses accordingly is called “in-context learning.” You will explore in-context learning with few-shot inference in the next section.

Examples 2-1, 2-2, and 2-3 show a restructured version of the previous chat example using the more complete prompt structure, including an instruction, and three prompt-completion examples in the context, followed by input data and the output indicator.

**Example 2-1. Instruction**

User: Answer the question using the format shown in the context.

**Example 2-2. Context**

Who won the baseball World Series in 2022?  
The Houston Astros won the World Series in 2022. They defeated the Philadelphia Phillies.

Who won the baseball World Series in 2021?  
The Atlanta Braves won the World Series in 2021. They defeated the Houston Astros.

Who won the baseball World Series in 2020?  
The Los Angeles Dodgers won the World Series in 2020. They defeated the Tampa Bay Rays.

**Example 2-3. Input data and output indicator**

Who won the baseball World Series in 2016?  
Assistant:

Let’s check the completion:

The Chicago Cubs won the World Series in 2016. They defeated the Cleveland Indians.

You can see how the model learned from the examples in the context and generated a completion in the desired format. Specifically, the assistant responded with a succinct answer that does not include extra details such as the final score of the baseball game—or the number of games in the series, as in the previous example.

The ideal prompt structure may vary depending on the task as well as the size of the model’s context window. The context window refers to the number of tokens the model can take as input when generating completions. Each model has a fixed context window size—anywhere from 512 tokens for FLAN-T5 to 100,000 tokens for Anthropic’s Claude model. For reference, Falcon has a context window size of 2,048 and Llama 2 has a context window size of 4,096. The context window size is often due to algorithmic limitations of the underlying neural network architecture. Also, in practice, you may see the model not fully utilizing a long sequence. This is often called “forgetting.” It’s important to test longer sequences and not assume the model will process 100,000 tokens the same way it would process an input of 1,000 tokens.

**Tip:** Some models document a single value: the maximum number of tokens. This number represents the combined total number of input tokens and generated output tokens.

The best prompt structure depends on how the generative model was trained and fine-tuned. Therefore, it’s important to read the documentation, specifically the model card, for a given generative model to gain intuition into the prompt structure used during training and tuning. Optimizing the prompt and prompt structure is all part of prompt engineering!

Next, you will learn how to further enrich the prompt context to evoke an emergent and thought-provoking property of generative AI models called in-context learning.

## In-Context Learning with Few-Shot Inference

A powerful technique to help your generative model produce better completions for your prompt is to include a few prompt-completion pairs inside the context portion of your prompt. This is called in-context learning with few-shot inference.

It’s worth noting that in-context learning does not modify the model in any way. The model adjusts—or learns—on the fly for the duration of that single request using the context provided in the prompt. This is a truly remarkable characteristic of generative models that can be used in many creative ways. You actually saw this in the previous human-assistant example that included a few examples, called shots, as part of the context.

In that case, adding just a few examples to the context helped the model respond with just the winner of the baseball World Series, without all of the other details generated in the examples before.

As you may have guessed, if you pass one prompt-completion pair into the context, this is called one-shot inference; if you pass no example at all, this is called zero-shot inference.

Zero-shot inference is often used to evaluate a model’s ability to perform a task that it hasn’t been explicitly trained on or seen examples for. For zero-shot inference, the model relies on its preexisting knowledge and generalization capabilities to make inferences or generate appropriate outputs, even when it encounters tasks or questions it has never seen before.

Larger models are typically able to capture more understanding of language (or perception, in the case of image-based models) and tend to generalize better to tasks they haven’t seen during their training phase. The largest models are surprisingly good at zero-shot inference and are able to infer and successfully complete many tasks that they were not specifically trained to perform.

In contrast, smaller models are generally only good at a small number of tasks, typically those that are similar to the task they were trained on. As part of the experiment and select step described in Chapter 1, you may have to try out a few models to find the right one for your use case. It’s worth noting that smaller models are typically faster during inference because of the reduced number of layers and parameters required for the computation.

Let’s look at examples of zero-shot, one-shot, and few-shot inference to help you visualize their differences in both prompt context and model response.

### Zero-Shot Inference

Here’s an example using zero prompts in the context.

**Prompt:**

User: Who won the 2016 baseball World Series?

**Completion:**

Assistant: The Chicago Cubs won the 2016 baseball World Series, defeating the Cleveland Indians in a thrilling seven-game series. It was the Cubs' first World Series title in 108 years, ending the longest championship drought in Major League Baseball history.

The zero-shot completion still includes additional information, because the model didn’t have any example to follow.

### One-Shot Inference

The following example adds an instruction and one-shot prompt in the context.

**Prompt:**

User: Answer the question using the format shown in the context.
Who won the baseball World Series in 2022?
The Houston Astros won the World Series in 2022. They defeated the Philadelphia Phillies.
Who won the 2016 baseball World Series?
Assistant:

**Completion:**

The 2016 World Series was won by the Chicago Cubs. They defeated the Cleveland Indians in seven games, becoming the first team to win the World Series after a 108-year drought.

The one-shot completion is closer to our desired output but still has a little bit too much detail.

### Few-Shot Inference

The following example shows what happens if you add few-shot prompt in the context.

**Prompt:**

User: Answer the question using the format shown in the context.
Who won the baseball World Series in 2022?
The Houston Astros won the World Series in 2022. They defeated the Philadelphia Phillies.

Who won the baseball World Series in 2021?
The Atlanta Braves won the World Series in 2021. They defeated the Houston Astros.

Who won the baseball World Series in 2020?
The Los Angeles Dodgers won the World Series in 2020. They defeated the Tampa Bay Rays.

Who won the 2016 baseball World Series?
Assistant:

**Completion:**

The Chicago Cubs won the World Series in 2016. They defeated the Cleveland Indians.

Note how the model generates a more focused and relevant response for just that request as you add more in-context examples called “shots.”

**Tip:** 2016 was a great year for one of the authors of this book, who is a lifelong Chicago Cubs fan!

There are subtle differences between the zero-shot, one-shot, and few-shot completions. With more examples, or shots, the model more closely follows the pattern of the response of the in-context prompt-completion pairs.

## In-Context Learning Gone Wrong

It’s worth noting that you can “trick” a model into temporarily learning an incorrect answer. For example, you can pass three in-context prompt-completion examples that demonstrate a positive customer review as a negative sentiment and a negative customer review as a positive sentiment.

Examples 2-4, 2-5, and 2-6 show a few-shot, in-context prompt with incorrect sentiment.

**Example 2-4. Instruction**

Classify the sentiment of the review using the format shown in the context.

**Example 2-5. Context**

Wrigley Field is a fun stadium to watch a baseball game on a day with a clear blue sky.  
Sentiment: NEGATIVE

The hungry seagulls during the 9th inning of a home Cubs game are very aggressive and annoying.  
Sentiment: POSITIVE

Murphy's is my favorite bar in Chicago on a Friday afternoon right before the start of a Cubs game.  
Sentiment: NEGATIVE

**Example 2-6. Input data and output indicator**

Harry Caray was the best Cubs sports announcer of all time!  
Sentiment:

**Completion:**

NEGATIVE

In this case, inference requests made to the model with this prompt are more likely to return the opposite sentiment. This is a peculiar but interesting quality of in-context learning, so it’s worth double-checking your in-context prompt-completion pairs carefully.

## In-Context Learning Best Practices

When you start experimenting with a new generative model, try zero-shot inference. If the model doesn’t generate the results you’re looking for, try one-shot and few-shot inference. For few-shot inference, it’s important to provide a consistent and appropriate mix of prompt-completion examples that represent the dataset and allow the model to properly learn from the provided context. Also, you should make sure that your context does not increase your prompt length above the input size or “context window” of the given generative model.

In-context learning is very useful, but the ability and limits for in-context learning vary across models. If you find yourself using upwards of five or six examples in your context and still not seeing the results you’re looking for, you may need to choose a different model or fine-tune an existing model. In Chapters 5, 6, and 7, you will explore various methods to fine-tune a foundational model.

In Chapter 9, you will see how to further augment the prompt using external data sources such as databases and knowledge stores. This is called retrieval-augmented generation (RAG) and is part of the larger generative AI ecosystem that helps augment prompts with domain knowledge. RAG improves model responses across many generative tasks and use cases.

Next, you’ll explore some prompt-engineering best practices to improve the responses from your generative AI models.

## Prompt-Engineering Best Practices

Constructing an effective prompt is both an art and a science. The following are some best practices to help you construct effective prompts for better generative results:

- **Be clear and concise.**  
  Prompts should be simple, straightforward, and avoid ambiguity. Clear prompts lead to more coherent responses. A general rule of thumb is this: if the wording is confusing to humans, it is likely to be confusing to these generative models. Simplify when possible.

- **Be creative.**  
  New and thought-provoking prompts can lead to unexpected, better, sometimes even innovative model completions.

- **Move the instruction to the end of the prompt for large amounts of text.**  
  If the context and input data are long, try moving the instruction to the end, right before the output indicator, as shown in the next example.

  **Prompt with small amount of input data and the instruction at the beginning:**

  Summarize the following conversation:

  Customer: How do I cancel my subscription account?  
  Support Representative: Click the cancel button at this link: ...  
  Customer: Thanks, that worked!

  **Summary:**

  **Prompt with long input data and the instruction at the end of the prompt:**

  Customer: Where is my order? It hasn't been delivered yet.  
  Support Representative: Can you give me your order ID, phone number, and last 4 digits of your credit card number?

  Long conversation between customer and support representative...

  Summarize the conversation above.

  **Summary:**

- **Clearly convey the subject.**  
  Clearly state one of the following: who, what, where, when, why, how, etc.

- **Use explicit directives.**  
  If you want the model to output in a particular format, specify that directly. For example, “Summarize the following customer-support dialog in a single sentence:”.

- **Avoid negative formulations.**  
  Negative formulations, while syntactically correct, may cause confusion. For example, use “Summarize in 5 sentences or less” instead of “Summarize in no more than 5 sentences.” Avoid negative formulations if a more straightforward linguistic variation exists.

- **Include context and few-shot example prompts.**  
  Provide additional context that helps the model respond more accurately. You can specify a single context across all inputs or a specific context for each input. You have already seen examples of including additional context in this chapter.

- **Specify the size of the response.**  
  Include the requested output size at the end of the prompt to focus the model. For example: “List the top 3 complaints from the following customer-support conversation:”

- **Provide a specific response format.**  
  Give the response format using an example. Include brackets for clarity. For example, “Summarize this document article in 10 words or less as shown here: [New generative AI model beats X benchmark by Y %.]”

- **Define what to do if the model can’t answer confidently.**  
  You can often ask the model to respond with, “I don’t know,” if it cannot confidently respond to the prompt. Otherwise, the model may generate a “hallucination” response, as shown in this example.

  **Prompt:**

  Which vehicle needs more down payment based on the following information:  
  The total cost of vehicle A is $40000 and it needs 30% as down payment.  
  The total cost of vehicle B is $50000 and it needs 20% as down payment.

  **Completion with an incorrect response (hallucination):**

  Based on the information provided, vehicle B requires a higher down payment than vehicle A. The down payment for vehicle B is $10000, which is $2000 higher than the down payment for vehicle A.  
  However, it is important to note that the down payment is not the only factor to consider when purchasing a vehicle.

  In this example, the model correctly calculated the down payment for vehicle B ($ 10,000), but incorrectly stated that it is $2,000 more than the down payment for vehicle A. The correct answer would be that down payment for vehicle A is higher ($12,000). The model didn’t know the right answer and just “hallucinated” a response.

  **Note:** While hallucinations are often fun to share with your coworkers during development, they are not fun to share with your end users in production!

  Let’s see how you can fix this. Here’s an example prompt that allows the model to respond with “I don’t know” in such cases.

  **Prompt:**

  Which vehicle needs more down payment based on the following information. If you are not confident, just answer "I don't know."  
  The total cost of vehicle A is $40000 and it needs 30% as down payment.  
  The total cost of vehicle B is $50000 and it needs 20% as down payment.

  **Completion:**

  I don't know.

  A more advanced prompting technique to address such scenarios is called chain-of-thought (CoT) prompting. CoT prompting shows the model how to reason through more complex tasks step-by-step to find the correct answer.

- **Ask the model to “think step-by-step.”**  
  If the model is confused about the instructions, you can ask the model to “think step-by-step,” which gives the model the freedom to break a single instruction into multiple steps.

  Here’s an example of a prompt using chain-of-thought (think step-by-step).

  **Prompt:**

  Which vehicle needs more down payment based on the following information (think step-by-step):  
  The total cost of vehicle A is $40000 and it needs 30% as down payment.  
  The total cost of vehicle B is $50000 and it needs 20% as down payment.

  **Completion:**

  Based on the information provided, vehicle A needs a higher down payment.

  The total cost of vehicle A is $40000 and it requires 30% down payment, which comes to $12000.  
  The total cost of vehicle B is $50000 and it requires 20% down payment, which comes to $10000.

  Therefore, vehicle A needs a higher down payment.

  Note how the model now reasoned through the task and generated the correct response.

  Depending on how they were trained and tuned, some models may respond to other variants of “think step-by-step” such as “divide into subtasks,” “approach the problem systematically,” “reason through the problem one step at a time,” etc.

  Some models may get confused when asked to perform a complex instruction that requires multiple steps. It’s important to recognize when the model is getting confused—and when you need to break the prompt down into multiple steps manually.

- **Add constraints for more control.**  
  Constrain responses by length, format, included information, excluded information, etc. For example, “Summarize this process in exactly 5 steps:”.

- **Evaluate the response.**  
  This seems obvious, but it’s worth noting that you should review the models’ responses to ensure the responses are high quality and appeal to your audience. Make changes to the prompts as needed.

  **Note:** Evaluating responses at scale is an open area of research. Human evaluation does not scale well, and automated evaluation may miss the nuances of human language. You will explore model evaluation in more detail in Chapter 5.

- **Use disclaimers or avoid prompts that the model should not answer.**  
  If your generative model is not equipped to respond to certain domains like law, medicine, or religion, you can instruct your model to respond with something like, “I am not licensed to provide medical advice. Please seek a licensed medical professional in your area.”

- **Use XML/HTML tags in your prompt.**  
  Some models support XML/HTML tags like `<tag>this is important</tag>` to create structure within the prompt. For example, if you want to reference an important piece of text in your input data, you can wrap that text in a tag to indicate where the important text starts and ends. You also ask some models to tag important parts of the response so you can parse the response and extract important data in a structured way.

- **Selectively focus.**  
  You can ask the model to only focus on certain parts of the input text. For example, you can ask that the model summarize only the first and last paragraph of your input data.

- **Mask personally identifiable information (PII).**  
  Some models can mask PII data in the response. You can ask these models to “remove all personally identifiable information and replace it with XXX,” for instance. Here is an example.

  **Prompt:**

  Human: Here is some text inside the `<text></text>` XML tags. Please remove all personally identifiable information from this text and replace it with XXX. It is very important that names, phone numbers, and email addresses are replaced with XXX.

  `<text>`  
  Joe: Hi Hannah!  
  Hannah: Hi Joe! Are you coming over?  
  Joe: Yup! Hey I, uh, forgot where you live."  
  Hannah: No problem! It's 1234 Wrightwood St, Chicago IL 60614.  
  Joe: Got it, thanks!  
  `</text>`

  Please put your sanitized version of the text with PII removed in `<response></response>` XML tags

  **Completion:**

  Assistant:  
  `<response>`  
  XXX: Hi XXX!  
  XXX: Hi XXX! Are you coming over?  
  XXX: Yup! Hey I, uh, forgot where you live."  
  XXX: No problem! It's XXX XXX, XXX XXX XXX XXX.  
  XXX: Got it, thanks!  
  `</response>`

By trying different prompts, and combining prompt engineering techniques, you see what works and what doesn’t work for your prompt, model, and use case combination. Continue to refine your prompt as needed. With more and more experimentation, you will gain the necessary intuition to quickly create and optimize a prompt to best suit your task and use case. Prompt engineering is an iterative skill that improves with practice, but prompt optimization is not as clear or well-studied as classical numerical optimization techniques, which you may find frustrating.

Take time to explore the creative and nondeterministic side of generative AI. At a minimum, you’ll enjoy a good laugh when the model surprises you with a seemingly random response to a question that you did not intend to ask.

Next, you will learn some common generative inference-specific parameters that influence the creativity of the generative model response. This is where the fun begins!

## Inference Configuration Parameters

Let’s examine configuration parameters to influence the way generative models generate text during inference. If you’ve used generative models in a “playground” such as Amazon SageMaker or Bedrock, you have likely seen slides and other numerical controls like the ones shown in Figure 2-1.

**Inference configuration parameters to control model outputs**  
*Figure 2-1. Inference configuration parameters to control model outputs*

These inference configuration parameters influence the model’s completion to your prompt. They give you fine-grained control over the length of the model response as well as the creativity. Each model exposes a different but often overlapping set of inference parameters. Often, these parameters are named similarly enough across models to reason through when you try out different models. Here are a few of the most common inference parameters:

- **Max new tokens**  
  This is one of the most obvious and straightforward parameters to tune. Use this parameter to limit the number of new tokens generated by the model. This is a very basic mechanism to keep model responses short and prevent rambling. Note that generating more tokens generally requires more computational resources and may result in longer inference times. Also note that reducing max new tokens is not a mechanism to prevent hallucinations; this may merely mask the hallucination by reducing its length.

- **Greedy versus random sampling**  
  During model inference, the model produces a probability distribution across all tokens in the model’s known vocabulary. The model chooses—or samples—a single token from this distribution as the next token to include in the response.

  For each inference request, you can configure the model to choose the next token using either greedy or random sampling. For greedy sampling, the token with the highest probability is selected. With random sampling, the model selects the next token using a random-weighted strategy across all predicted token probabilities. The different sampling methods are shown in Figure 2-2 for the phrase “the student learns from the professor and her lectures.”

  Most generative model-inference implementations default to greedy sampling, also called greedy decoding. This is the simplest form of next-token prediction, as the model always chooses the word with the highest probability. This method works well for very short generations but may result in repeated tokens or sequences of tokens.

  If you want to generate text that is more natural and minimizes repeating tokens, you can configure the model to use random sampling during inference. This will cause the model to randomly choose the next token using a weighted strategy across the probability distribution. The token student, as shown here, has a probability score of 0.02. With random sampling, this equates to a 2% chance that this word will be selected from the distribution.

  **Greedy versus random sampling to predict the next token from a probability distribution**  
  *Figure 2-2. Greedy versus random sampling to predict the next token from a probability distribution*

  Using random sampling, you reduce the likelihood of repeated tokens in the model completion. The trade-off, however, is that the model output may be too creative and either generate an off-topic or unintelligible response. The challenge of finding this optimal setting is why this is called prompt engineering!

  **Tip:** Some libraries like Hugging Face Transformers may require you to explicitly disable greedy sampling and manually enable random sampling using a function argument similar to `do_sample=True`.

- **top-p and top-k random sampling**  
  These are the most common inference parameters when using random sampling. These parameters provide more fine-grained control for the random sample, which, if used properly, should improve the model’s response while allowing it to be creative enough to fulfill the generative task.

  **top-k**, as you may have guessed, limits the model to choosing a token randomly from only the top-k tokens with the highest probability. For example, if k is set to 3, you are restricting the model to choose from only the top three tokens using the weighted random-sampling strategy. In this case, the model randomly chooses “from” as the next token, although it could have selected one of the other two, as shown in Figure 2-3.

  **In this case top-k sampling restricts the model to choose from the top 3 probabilities.**  
  *Figure 2-3. In this case, top-k sampling restricts the model to choosing from the top three probabilities*

  Note that setting top-k to a higher number can help reduce repetitiveness, while setting top-k to 1 basically gives you greedy decoding.

  **top-p** limits the model to randomly sampling from the set of tokens whose cumulative probabilities do not exceed p, starting from the highest probability and working down to the lowest probability. To illustrate this, first sort the tokens in descending order based on the probability. Then select a subset of tokens whose cumulative probability scores do not exceed p.

  For example, if p = 0.32, the options are “learns”, “from”, and “student” since their probabilities of 0.20, 0.10, and 0.02, respectively, add up to 0.32. The model then uses the weighted random-sampling strategy to choose the next token—“student” in this case—from this subset of tokens, as shown in Figure 2-4.

  **top-p can also produce greater variability and is sometimes used if it is hard to pick a good top-k value. top-p and top-k can also be used together.**

  **Top p random probability weighting**  
  *Figure 2-4. top-p random probability weighting*

- **temperature**  
  This parameter also helps to control the randomness of the model output by modifying the shape of the next-token probability distribution. In general, the higher the temperature, the higher the randomness; the lower the temperature, the lower the randomness.

  In contrast to top-k and top-p, changing the temperature actually changes the next-token probability distribution, which ultimately affects the next-token prediction.

  A low temperature (below 1, for example) results in stronger peaks where the probabilities are concentrated among a smaller subset of tokens. A higher temperature (above 1, for example) results in a flatter next-token probability distribution where the probabilities are more evenly spread across the tokens. Setting the temperature to 1 leaves the next-token probability distribution unaltered, which represents the distribution learned during model training and tuning.

  **Changing the temperature will change the next token probability distribution.**  
  *Figure 2-5. Changing the temperature will change the next-token probability distribution*

  In both cases, the model selects the next token from the modified probability distribution using either greedy or random sampling, which is orthogonal to the temperature parameter.

  Note that if the temperature value is too low, the model may generate more repetitions; if the temperature is too high, the model may generate nonsensical output. However, starting with a temperature value of 1 is usually a good strategy.

## Summary

In this chapter, you learned techniques to help get the best possible performance from generative AI models using prompt engineering and by experimenting with different inference configuration parameters. Prompt engineering guides the generative foundation model to provide more relevant and accurate completions using various methods such as better-worded prompts, in-context learning examples, and step-by-step logical reasoning.

While you can get far with prompt engineering, in-context learning, and inference parameters, these techniques do not actually modify the generative models’ weights. As such, you may need to train or fine-tune a generative model on your own datasets to help it better understand your specific domain and set of generative use cases, which you will explore in the next few chapters.