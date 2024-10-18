Chapter 11
Controlled Generation and Fine-Tuning with Stable Diffusion
This chapter focuses on controlling image generation in diffusion models such as Stable Diffusion by introducing techniques that allow for greater control and customization. These methods are critical for tasks such as edge detection, segmentation, and fine-tuning models for personalized content creation. You will also explore reinforcement learning for aligning model behavior with human preferences, making the generated content more aligned with ethical guidelines and user expectations.

ControlNet
ControlNet is a deep neural network introduced in 2023, designed to augment diffusion models by adding specific control mechanisms. These controls allow you to guide image generation with predefined tasks, such as edge detection, depth mapping, or hand-drawn scribbles. Each control learns from input data and enhances the output by guiding Stable Diffusion's generation process.

ControlNet works by conditioning image generation based on external control inputs. For example, if you provide an edge-detected version of an image, ControlNet can guide Stable Diffusion to generate an output that respects those edges while adhering to a new prompt. The base image and control outputs, like edge maps, depth maps, or boundary detection, are fed into Stable Diffusion alongside a new textual prompt to produce a custom output.

Table 11-1 provides examples of control outputs, such as Canny edge maps or depth maps, that guide image generation. These controls give users creative freedom to maintain specific details from a base image, even when generating entirely new content with new prompts.

Example: Canny Edge Control
The Canny edge detection technique extracts edges from an image and uses that data to guide image generation. For example, the base image can be processed through a Canny edge map, which is then input into Stable Diffusion to generate a new image based on a user prompt while maintaining the structure of the edges.


    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    import torch

    # Load the control model
    canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

    # Initialize the Stable Diffusion pipeline with ControlNet
    sd_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=canny, torch_dtype=torch.float16
    )

    generator = torch.manual_seed(0)

    # Use the pipeline to generate an image with a Canny edge map control
    out_image = sd_pipe(
        "metal orange colored car, complete car, color photo, outdoors in a pleasant landscape, realistic, high quality", 
        num_inference_steps=20, generator=generator, image=canny_image
    ).images[0]
    
In this example, the Canny edge map was generated from a base image and then used as a guide to generate an orange sports car in a new landscape. This method offers powerful ways to generate creative yet controlled images.

Fine-Tuning Techniques
Fine-tuning is essential for customizing image generation models like Stable Diffusion to include new concepts or subjects not covered in the original training dataset. The chapter introduces several fine-tuning techniques:

DreamBooth
DreamBooth is a fine-tuning method that allows models to learn new concepts from just a few images. It was introduced in 2023 and can be used to personalize text-to-image models with as few as 3-5 sample images. DreamBooth’s applications range from creative content generation to marketing. For example, you can fine-tune the model to generate images of a new product in different environments or artistic styles. Figure 11-5 shows an example where DreamBooth was used to generate images of a dog, “Molly,” in various contexts.

DreamBooth enables you to generate images of an object, product, or pet in settings the object has never been in before, offering broad applications in content creation. It also supports tasks like art rendition (generating images in the style of famous painters) and text-guided view synthesis (changing perspectives or viewpoints).

PEFT and LoRA
Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA) offer lightweight alternatives to full fine-tuning. Instead of retraining the entire model, PEFT and LoRA allow you to fine-tune specific components, such as cross-attention layers. This results in significantly smaller model adapters (between 2 to 500 MB), which are easier to deploy and maintain.

For instance, by using LoRA with DreamBooth, you can fine-tune only the cross-attention layers responsible for aligning the text and image data. The result is a fine-tuned model that generates custom images without requiring the memory and computational resources of a fully fine-tuned model.

Textual Inversion
Textual Inversion is another fine-tuning technique that allows personalization without modifying the entire model. It works by creating new embeddings for custom tokens, which are used to represent new concepts. For example, you can define a pseudoword (e.g., "M*") and use it to refer to a custom object or style in your prompt. Textual inversion learns to associate this pseudoword with the subject in the fine-tuned images.

In this case, the model is trained to recognize the concept of "Molly" as represented by the pseudoword "M*." During inference, users can refer to "M*" in the prompt to generate images of the concept without explicitly mentioning the subject by name. This technique allows you to fine-tune with very few images and without overloading the model with new data.

Reinforcement Learning from Human Feedback (RLHF)
Reinforcement Learning from Human Feedback (RLHF) is a technique used to align models with human preferences. Similar to the RLHF process in large language models, diffusion models can also be fine-tuned using human feedback to improve aesthetics, compressibility, and content moderation. RLHF applies to multimodal models to ensure generated content meets certain ethical standards.

An example of this process involves training the model to generate aesthetically appealing images using the LAION-Aesthetics dataset, which contains human-labeled ratings of image quality. The model is fine-tuned to maximize rewards based on human preferences, resulting in outputs that align with user expectations.

Content Moderation with Amazon Rekognition
In addition to improving image quality, RLHF can be used for content moderation. By using Amazon Rekognition's Content Moderation API, you can fine-tune the diffusion model to flag inappropriate or harmful content. The API provides detailed labels with confidence scores, allowing the model to recognize offensive or unwanted imagery and respond appropriately.


    {
    "ModerationLabels": [
        {
            "Confidence": 99.2,
            "ParentName": "",
            "Name": "Visually Disturbing"
        },
        {
            "Confidence": 99.2,
            "ParentName": "Visually Disturbing",
            "Name": "Air Crash"
        }
    ]
    }
    
Conclusion
In this chapter, you learned how to control image generation using techniques like ControlNet, fine-tune models with custom datasets using DreamBooth and textual inversion, and apply reinforcement learning to align generated content with human preferences. These advanced techniques provide flexibility in content creation, allowing models to generate images that meet user-defined criteria while adhering to ethical standards.

The next chapter will explore how to leverage Amazon Bedrock for managing generative AI tasks and using these tools in a production environment.

References: Lvmin Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models," arXiv, 2023. Nataniel Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation," arXiv, 2023. Rinon Gal et al., "An Image Is Worth One Word: Personalizing Text-to-Image Generation Using Textual Inversion," arXiv, 2022.
