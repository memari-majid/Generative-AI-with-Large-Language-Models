Chapter 6: Parameter-Efficient Fine-Tuning (PEFT)
In this chapter, you explored Parameter-Efficient Fine-Tuning (PEFT), which offers a solution to the high computational and memory demands of full model fine-tuning. Rather than updating all parameters of a large generative model, PEFT techniques enable the fine-tuning of a small subset of parameters, which drastically reduces resource requirements while still maintaining reasonable performance.

Key Concepts of PEFT
Traditional fine-tuning updates every parameter in a model, requiring substantial memory and compute resources. PEFT techniques, by contrast, freeze the majority of the model’s parameters and only modify a small number, often just 1–2% of the total parameters. This reduces memory consumption, enables fine-tuning with limited hardware (e.g., a single GPU), and prevents "catastrophic forgetting" since the original model weights remain unchanged. PEFT is particularly useful for domain adaptation, task-specific tuning, and multi-tenant systems where the model must be adapted to handle different tasks with minimal cost.

Comparison: Full Fine-Tuning vs. PEFT
The main difference between full fine-tuning and PEFT is the number of model parameters that are updated during training:

Full Fine-Tuning: Updates all model parameters, leading to high memory requirements (optimizer states, gradients, activations) and significant compute resources. This can quickly exhaust available GPU memory, especially for large models like LLMs.
PEFT: Updates only a subset of the model parameters, leaving the original foundation model frozen. As a result, it reduces memory and compute requirements, making fine-tuning feasible even on hardware with limited resources. This can be done with smaller datasets and is useful when there is a need to fine-tune a model for multiple tenants or tasks without creating multiple full copies of the model.
PEFT Techniques
There are two main types of PEFT techniques: additive and reparameterization methods:

Additive Techniques (e.g., Prompt Tuning): These methods augment the model by adding extra layers or parameters to the pretrained model. In prompt tuning, trainable tokens are prepended to the input prompts to optimize the model's responses for specific tasks.
Reparameterization Techniques (e.g., LoRA, QLoRA): These techniques involve training additional low-rank matrices or using quantization. The pretrained model’s parameters remain frozen, while smaller, low-rank matrices are introduced to efficiently adapt the model. This results in far fewer parameters to train and much lower memory usage compared to full fine-tuning.
Low-Rank Adaptation (LoRA)
LoRA (Low-Rank Adaptation) is a widely used reparameterization method that modifies only a small portion of the model by introducing low-rank matrices into the Transformer architecture's linear layers. These matrices, represented as A and B, are much smaller than the original weight matrices, which significantly reduces the number of parameters that need to be trained.

For example, in the case of a matrix with 32,768 trainable parameters, LoRA introduces two smaller matrices with dimensions 4 × 64 (256 parameters) and 512 × 4 (2,048 parameters), resulting in only 2,304 parameters to fine-tune—dramatically reducing resource consumption. The original weights remain frozen, and the updated low-rank matrices are combined with the original model weights during inference.

Quantized LoRA (QLoRA)
QLoRA builds on LoRA by incorporating quantization to further reduce memory usage. In QLoRA, model weights are stored in a 4-bit format using a technique called double quantization, where the 4-bit weights are dequantized to 16 bits during forward and backward passes. This allows the model to use even less memory while maintaining competitive performance.

The QLoRA technique is particularly useful for fine-tuning on resource-constrained devices and achieves performance similar to 16-bit fine-tuning, making it highly efficient for large language models.

Prompt Tuning
Prompt tuning, another PEFT method, differs from prompt engineering. Rather than manually crafting prompts, prompt tuning adds trainable soft tokens to the input prompt. These tokens are virtual and do not correspond to natural language but represent vectors in the model's embedding space. The goal of prompt tuning is to optimize these soft tokens to improve task-specific performance.

Unlike LoRA, prompt tuning does not modify the model weights; instead, it optimizes the input instructions. Prompt tuning works best with larger models and can achieve performance similar to full fine-tuning for models with billions of parameters. However, its primary drawback is interpretability since the soft tokens do not directly correspond to natural language tokens.

Considerations for Choosing PEFT Over Full Fine-Tuning
When deciding between full fine-tuning and PEFT, several factors should be considered:

Compute and Memory Requirements: Full fine-tuning requires significantly more resources, while PEFT drastically reduces the compute and memory needed.
Task-Specific Adaptation: PEFT is highly efficient for adapting models to multiple tasks or tenants without duplicating the full model.
Performance Trade-offs: While full fine-tuning typically offers higher performance, PEFT methods like LoRA and QLoRA can achieve similar performance with far fewer parameters and lower cost. In some cases, the performance difference is minimal.
Performance Comparison: Full Fine-Tuning vs. LoRA
Performance can be compared using evaluation metrics like ROUGE. Full fine-tuning typically provides the highest performance, but LoRA and other PEFT techniques perform similarly, with only a small reduction in accuracy. For example, in a dialogue summarization task, full fine-tuning might yield a ROUGE-1 score of 0.4216, while LoRA-based fine-tuning achieves a score of 0.4081—a minor difference, but with significantly fewer resources.

Summary
PEFT techniques, including LoRA, QLoRA, and prompt tuning, allow for efficient fine-tuning of large language models by significantly reducing the number of trainable parameters. LoRA and QLoRA are particularly effective for fine-tuning models with limited computational resources, while prompt tuning optimizes input instructions for specific tasks. These methods provide a valuable alternative to full fine-tuning, offering a balance between performance and resource efficiency.
