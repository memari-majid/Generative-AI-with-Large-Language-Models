#!/usr/bin/env python
# coding: utf-8

# # Textual-inversion fine-tuning for Stable Diffusion using d🧨ffusers 
# 
# This notebook shows how to "teach" Stable Diffusion a new concept via textual-inversion using 🤗 Hugging Face [🧨 Diffusers library](https://github.com/huggingface/diffusers). 
# 
# ![Textual Inversion example](https://textual-inversion.github.io/static/images/editing/colorful_teapot.JPG)
# _By using just 3-5 images you can teach new concepts to Stable Diffusion and personalize the model on your own images_ 

# ## Initial setup

# In[8]:


%pip install -U torch==2.0.1 \
  torchvision==0.15.2 \
  triton==2.0.0 \
  accelerate==0.23.0 \
  transformers==4.33.2 \
  ftfy==6.1.1 \
  git+https://github.com/huggingface/diffusers@a0198676d7eb73ebb1e409cf90bd963308025423

#  diffusers==0.21.2

# In[9]:


import argparse
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version #, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


# if is_wandb_available():
#     import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
#check_min_version("0.22.0.dev0")

logger = get_logger(__name__)

# In[10]:


def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()
    return images


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


# In[11]:


parser = argparse.ArgumentParser()

parser.add_argument(
    "--save_steps",
    type=int,
    default=500,
    help="Save learned_embeds.bin every X updates steps.",
)
parser.add_argument(
    "--save_as_full_pipeline",
    action="store_true",
    help="Save the complete stable diffusion pipeline.",
)
parser.add_argument(
    "--num_vectors",
    type=int,
    default=1,
    help="How many textual inversion vectors shall be used to learn the concept.",
)
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default="stabilityai/stable-diffusion-xl-base-1.0",
#    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--revision",
    type=str,
    default=None,
    required=False,
    help="Revision of pretrained model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default=None,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--train_data_dir", 
    type=str, 
    default="images/", 
#    required=True, 
    help="A folder containing the training data."
)
parser.add_argument(
    "--placeholder_token",
    type=str,
    default="M*",
#    required=True,
    help="A token to use as a placeholder for the concept.",
)
parser.add_argument(
    "--initializer_token", 
    type=str, 
    default="dog", 
    required=False, 
    help="A token to use as initializer word."
)
parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="text-inversion-model",
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
parser.add_argument(
    "--resolution",
    type=int,
    default=512,
    help=(
        "The resolution for input images, all the images in the train/validation dataset will be resized to this"
        " resolution"
    ),
)
parser.add_argument(
    "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
)
parser.add_argument(
    "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
)
parser.add_argument("--num_train_epochs", type=int, default=100)
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=200,
    help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--gradient_checkpointing",
    action="store_true",
    help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--scale_lr",
    action="store_true",
    default=False,
    help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
)
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="constant",
    help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
    ),
)
parser.add_argument(
    "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
)
parser.add_argument(
    "--lr_num_cycles",
    type=int,
    default=1,
    help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
)
parser.add_argument(
    "--dataloader_num_workers",
    type=int,
    default=0,
    help=(
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
    ),
)
parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
# parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
# parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
# parser.add_argument(
#     "--hub_model_id",
#     type=str,
#     default=None,
#     help="The name of the repository to keep in sync with the local `output_dir`.",
# )
parser.add_argument(
    "--logging_dir",
    type=str,
    default="logs",
    help=(
        "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
        " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
    ),
)
parser.add_argument(
    "--mixed_precision",
    type=str,
    default="no",
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU."
    ),
)
parser.add_argument(
    "--allow_tf32",
    action="store_true",
    help=(
        "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
        " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
    ),
)
parser.add_argument(
    "--report_to",
    type=str,
    default="tensorboard",
    help=(
        'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
        ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
    ),
)
parser.add_argument(
    "--validation_prompt",
    type=str,
    default=None,
    help="A prompt that is used during validation to verify that the model is learning.",
)
parser.add_argument(
    "--num_validation_images",
    type=int,
    default=4,
    help="Number of images that should be generated during validation with `validation_prompt`.",
)
parser.add_argument(
    "--validation_steps",
    type=int,
    default=100,
    help=(
        "Run validation every X steps. Validation consists of running the prompt"
        " `args.validation_prompt` multiple times: `args.num_validation_images`"
        " and logging the images."
    ),
)
parser.add_argument(
    "--validation_epochs",
    type=int,
    default=None,
    help=(
        "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
        " `args.validation_prompt` multiple times: `args.num_validation_images`"
        " and logging the images."
    ),
)
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument(
    "--checkpointing_steps",
    type=int,
    default=500,
    help=(
        "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
        " training using `--resume_from_checkpoint`."
    ),
)
parser.add_argument(
    "--checkpoints_total_limit",
    type=int,
    default=None,
    help=("Max number of checkpoints to store."),
)
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help=(
        "Whether training should be resumed from a previous checkpoint. Use a path saved by"
        ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
    ),
)
parser.add_argument(
    "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
)
parser.add_argument(
    "--no_safe_serialization",
    action="store_true",
    help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
)

args, _ = parser.parse_known_args()

env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
if env_local_rank != -1 and env_local_rank != args.local_rank:
    args.local_rank = env_local_rank

if args.train_data_dir is None:
    raise ValueError("You must specify a train data directory.")

print(args)

# In[12]:


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

# In[13]:


logging_dir = os.path.join(args.output_dir, args.logging_dir)
accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
)

# if args.report_to == "wandb":
#     if not is_wandb_available():
#         raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

# Handle the repository creation
if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

#     if args.push_to_hub:
#         repo_id = create_repo(
#             repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
#         ).repo_id

# Load tokenizer
if args.tokenizer_name:
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
elif args.pretrained_model_name_or_path:
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
)

# Add the placeholder token in tokenizer
placeholder_tokens = [args.placeholder_token]

if args.num_vectors < 1:
    raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

# add dummy tokens for multi-vector
additional_tokens = []
for i in range(1, args.num_vectors):
    additional_tokens.append(f"{args.placeholder_token}_{i}")
placeholder_tokens += additional_tokens

num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
if num_added_tokens != args.num_vectors:
    raise ValueError(
        f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
        " `placeholder_token` that is not already in the tokenizer."
    )

# Convert the initializer_token, placeholder_token to ids
token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
# Check if initializer_token is a single token or a sequence of tokens
if len(token_ids) > 1:
    raise ValueError("The initializer token must be a single token.")

initializer_token_id = token_ids[0]
placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

# Resize the token embeddings as we are adding new special tokens to the tokenizer
text_encoder.resize_token_embeddings(len(tokenizer))

# Initialise the newly added placeholder token with the embeddings of the initializer token
token_embeds = text_encoder.get_input_embeddings().weight.data
with torch.no_grad():
    for token_id in placeholder_token_ids:
        token_embeds[token_id] = token_embeds[initializer_token_id].clone()

# Freeze vae and unet
vae.requires_grad_(False)
unet.requires_grad_(False)
# Freeze all parameters except for the token embeddings in text encoder
text_encoder.text_model.encoder.requires_grad_(False)
text_encoder.text_model.final_layer_norm.requires_grad_(False)
text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

if args.gradient_checkpointing:
    # Keep unet in train mode if we are using gradient checkpointing to save memory.
    # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
    unet.train()
    text_encoder.gradient_checkpointing_enable()
    unet.enable_gradient_checkpointing()

if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

# Enable TF32 for faster training on Ampere GPUs,
# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

if args.scale_lr:
    args.learning_rate = (
        args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    )

# Initialize the optimizer
optimizer = torch.optim.AdamW(
    text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

# Dataset and DataLoaders creation:
train_dataset = TextualInversionDataset(
    data_root=args.train_data_dir,
    tokenizer=tokenizer,
    size=args.resolution,
    placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
    repeats=args.repeats,
    learnable_property=args.learnable_property,
    center_crop=args.center_crop,
    set="train",
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
)
if args.validation_epochs is not None:
    warnings.warn(
        f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
        " Deprecated validation_epochs in favor of `validation_steps`"
        f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
        FutureWarning,
        stacklevel=2,
    )
    args.validation_steps = args.validation_epochs * len(train_dataset)

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    num_training_steps=args.max_train_steps * accelerator.num_processes,
    num_cycles=args.lr_num_cycles,
)

# Prepare everything with our `accelerator`.
text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    text_encoder, optimizer, train_dataloader, lr_scheduler
)

# For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
# as these weights are only used for inference, keeping weights in full precision is not required.
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

# Move vae and unet to device and cast to weight_dtype
unet.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)

# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

# We need to initialize the trackers we use, and also store our configuration.
# The trackers initializes automatically on the main process.
if accelerator.is_main_process:
    accelerator.init_trackers("textual_inversion", config=vars(args))

# Train!
total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")
global_step = 0
first_epoch = 0

# Potentially load in the weights and states from a previous save
if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

# Only show the progress bar once on each machine.
progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
progress_bar.set_description("Steps")

# keep original embeddings as reference
orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

for epoch in range(first_epoch, args.num_train_epochs):
    text_encoder.train()
    for step, batch in enumerate(train_dataloader):
        # Skip steps until we reach the resumed step
        if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            if step % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
            continue

        with accelerator.accumulate(text_encoder):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

            with torch.no_grad():
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            images = []
            progress_bar.update(1)
            global_step += 1
            if global_step % args.save_steps == 0:
                weight_name = (
                    f"learned_embeds-steps-{global_step}.bin"
                    if args.no_safe_serialization
                    else f"learned_embeds-steps-{global_step}.safetensors"
                )
                save_path = os.path.join(args.output_dir, weight_name)
                save_progress(
                    text_encoder,
                    placeholder_token_ids,
                    accelerator,
                    args,
                    save_path,
                    safe_serialization=not args.no_safe_serialization,
                )

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    images = log_validation(
                        text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch
                    )

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break
            

# Create the pipeline using the trained modules and save it.
accelerator.wait_for_everyone()

save_full_model = args.save_as_full_pipeline

if save_full_model:
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
    )
    pipeline.save_pretrained(args.output_dir)
# Save the newly trained embeddings
weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
save_path = os.path.join(args.output_dir, weight_name)
save_progress(
    text_encoder,
    placeholder_token_ids,
    accelerator,
    args,
    save_path,
    safe_serialization=not args.no_safe_serialization,
)

accelerator.end_training()

# In[ ]:






# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:


# #@title [Optional] Install xformers for faster and memory efficient training
# #@markdown Acknowledgement: The xformers wheel are taken from [TheLastBen/fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion). Thanks a lot for building these wheels!


# from subprocess import getoutput
# from IPython.display import HTML
# from IPython.display import clear_output
# import time

# s = getoutput('nvidia-smi')
# if 'T4' in s:
#   gpu = 'T4'
# elif 'P100' in s:
#   gpu = 'P100'
# elif 'V100' in s:
#   gpu = 'V100'
# elif 'A100' in s:
#   gpu = 'A100'

# while True:
#     try: 
#         gpu=='T4'or gpu=='P100'or gpu=='V100'or gpu=='A100'
#         break
#     except:
#         pass
#     print('[1;31mit seems that your GPU is not supported at the moment')
#     time.sleep(5)

# if (gpu=='T4'):
#   %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/T4/xformers-0.0.13.dev0-py3-none-any.whl
  
# elif (gpu=='P100'):
#   %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/P100/xformers-0.0.13.dev0-py3-none-any.whl

# elif (gpu=='V100'):
#   %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/V100/xformers-0.0.13.dev0-py3-none-any.whl

# elif (gpu=='A100'):
#   %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/A100/xformers-0.0.13.dev0-py3-none-any.whl

# In[ ]:


# #@title [Optional] Login to the Hugging Face Hub
# #@markdown Add a token with the "Write Access" role to be able to add your trained concept to the [Library of Concepts](https://huggingface.co/sd-concepts-library)
# from huggingface_hub import notebook_login

# notebook_login()

# In[ ]:


# #@title Import required libraries
# import argparse
# import itertools
# import math
# import os
# import random

# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.utils.checkpoint
# from torch.utils.data import Dataset

# import PIL
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import set_seed
# from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
# from diffusers.optimization import get_scheduler
# from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
# from torchvision import transforms
# from tqdm.auto import tqdm
# from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# ## Settings for teaching your new concept

# In[ ]:


# #@markdown `pretrained_model_name_or_path` which Stable Diffusion checkpoint you want to use
# pretrained_model_name_or_path = "stabilityai/stable-diffusion-2" #@param ["stabilityai/stable-diffusion-2", "stabilityai/stable-diffusion-2-base", "CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5"] {allow-input: true}

# ### Get the training images:

# #### Download the images from the internet and save them locally.
# 
# You can also upload the images to colab or load from google drive, please check the next section if you want to use that.

# In[ ]:


# #@markdown Add here the URLs to the images of the concept you are adding. 3-5 should be fine
# urls = [
#       "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
#       "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
#       "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
#       "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
#       ## You can add additional images here
# ]

# In[ ]:


# #@title Download
# import requests
# import glob
# from io import BytesIO

# def download_image(url):
#   try:
#     response = requests.get(url)
#   except:
#     return None
#   return Image.open(BytesIO(response.content)).convert("RGB")

# images = list(filter(None,[download_image(url) for url in urls]))
# save_path = "./my_concept"
# if not os.path.exists(save_path):
#   os.mkdir(save_path)
# [image.save(f"{save_path}/{i}.jpeg") for i, image in enumerate(images)]

# #### Load images from local folder or google drive
# 
# You can also load your own training images from google drive or upload them to colab usingthe files taband then provide the path to the directory containing images. 
# 
# *Make sure that the directory only contains images as the following cells will read all the files from the provided directory.*

# In[ ]:


# from google.colab import drive
# drive.mount('/content/gdrive')

# In[ ]:


# #@markdown `images_path` is a path to directory containing the training images. It could 
# images_path = "" #@param {type:"string"}
# while not os.path.exists(str(images_path)):
#   print('The images_path specified does not exist, use the file explorer to copy the path :')
#   images_path=input("")
# save_path = images_path

# ####  Setup and check the images you have just added

# In[ ]:


# save_path = './images'

# In[ ]:


# images = []
# for file_path in os.listdir(save_path):
#   try:
#       image_path = os.path.join(save_path, file_path)
#       images.append(Image.open(image_path).resize((512, 512)))
#   except:
#     print(f"{image_path} is not a valid image, please make sure to remove this file from the directory otherwise the training could fail.")
# image_grid(images, 1, len(images))

# In[ ]:


# #@title Settings for your newly created concept
# #@markdown `what_to_teach`: what is it that you are teaching? `object` enables you to teach the model a new object to be used, `style` allows you to teach the model a new style one can use.
# what_to_teach = "object" #@param ["object", "style"]
# #@markdown `placeholder_token` is the token you are going to use to represent your new concept (so when you prompt the model, you will say "A `<my-placeholder-token>` in an amusement park"). We use angle brackets to differentiate a token from other words/tokens, to avoid collision.
# placeholder_token = "<cat-toy>" #@param {type:"string"}
# #@markdown `initializer_token` is a word that can summarise what your new concept is, to be used as a starting point
# initializer_token = "toy" #@param {type:"string"}

# ## Teach the model a new concept (fine-tuning with textual inversion)
# Execute this this sequence of cells to run the training process. The whole process may take from 1-4 hours. (Open this block if you are interested in how this process works under the hood or if you want to change advanced training settings or hyperparameters)

# ### Create Dataset

# In[ ]:


# #@title Setup the prompt templates for training 
# imagenet_templates_small = [
#     "a photo of a {}",
#     "a rendering of a {}",
#     "a cropped photo of the {}",
#     "the photo of a {}",
#     "a photo of a clean {}",
#     "a photo of a dirty {}",
#     "a dark photo of the {}",
#     "a photo of my {}",
#     "a photo of the cool {}",
#     "a close-up photo of a {}",
#     "a bright photo of the {}",
#     "a cropped photo of a {}",
#     "a photo of the {}",
#     "a good photo of the {}",
#     "a photo of one {}",
#     "a close-up photo of the {}",
#     "a rendition of the {}",
#     "a photo of the clean {}",
#     "a rendition of a {}",
#     "a photo of a nice {}",
#     "a good photo of a {}",
#     "a photo of the nice {}",
#     "a photo of the small {}",
#     "a photo of the weird {}",
#     "a photo of the large {}",
#     "a photo of a cool {}",
#     "a photo of a small {}",
# ]

# imagenet_style_templates_small = [
#     "a painting in the style of {}",
#     "a rendering in the style of {}",
#     "a cropped painting in the style of {}",
#     "the painting in the style of {}",
#     "a clean painting in the style of {}",
#     "a dirty painting in the style of {}",
#     "a dark painting in the style of {}",
#     "a picture in the style of {}",
#     "a cool painting in the style of {}",
#     "a close-up painting in the style of {}",
#     "a bright painting in the style of {}",
#     "a cropped painting in the style of {}",
#     "a good painting in the style of {}",
#     "a close-up painting in the style of {}",
#     "a rendition in the style of {}",
#     "a nice painting in the style of {}",
#     "a small painting in the style of {}",
#     "a weird painting in the style of {}",
#     "a large painting in the style of {}",
# ]

# In[ ]:


# #@title Setup the dataset
# class TextualInversionDataset(Dataset):
#     def __init__(
#         self,
#         data_root,
#         tokenizer,
#         learnable_property="object",  # [object, style]
#         size=512,
#         repeats=100,
#         interpolation="bicubic",
#         flip_p=0.5,
#         set="train",
#         placeholder_token="*",
#         center_crop=False,
#     ):

#         self.data_root = data_root
#         self.tokenizer = tokenizer
#         self.learnable_property = learnable_property
#         self.size = size
#         self.placeholder_token = placeholder_token
#         self.center_crop = center_crop
#         self.flip_p = flip_p

#         self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

#         self.num_images = len(self.image_paths)
#         self._length = self.num_images

#         if set == "train":
#             self._length = self.num_images * repeats

#         self.interpolation = {
#             "linear": PIL.Image.LINEAR,
#             "bilinear": PIL.Image.BILINEAR,
#             "bicubic": PIL.Image.BICUBIC,
#             "lanczos": PIL.Image.LANCZOS,
#         }[interpolation]

#         self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
#         self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

#     def __len__(self):
#         return self._length

#     def __getitem__(self, i):
#         example = {}
#         image = Image.open(self.image_paths[i % self.num_images])

#         if not image.mode == "RGB":
#             image = image.convert("RGB")

#         placeholder_string = self.placeholder_token
#         text = random.choice(self.templates).format(placeholder_string)

#         example["input_ids"] = self.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.tokenizer.model_max_length,
#             return_tensors="pt",
#         ).input_ids[0]

#         # default to score-sde preprocessing
#         img = np.array(image).astype(np.uint8)

#         if self.center_crop:
#             crop = min(img.shape[0], img.shape[1])
#             h, w, = (
#                 img.shape[0],
#                 img.shape[1],
#             )
#             img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

#         image = Image.fromarray(img)
#         image = image.resize((self.size, self.size), resample=self.interpolation)

#         image = self.flip_transform(image)
#         image = np.array(image).astype(np.uint8)
#         image = (image / 127.5 - 1.0).astype(np.float32)

#         example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
#         return example

# ### Setting up the model

# In[ ]:


# #@title Load the tokenizer and add the placeholder token as a additional special token.
# tokenizer = CLIPTokenizer.from_pretrained(
#     pretrained_model_name_or_path,
#     subfolder="tokenizer",
# )

# # Add the placeholder token in tokenizer
# num_added_tokens = tokenizer.add_tokens(placeholder_token)
# if num_added_tokens == 0:
#     raise ValueError(
#         f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
#         " `placeholder_token` that is not already in the tokenizer."
#     )

# In[ ]:


# #@title Get token ids for our placeholder and initializer token. This code block will complain if initializer string is not a single token
# # Convert the initializer_token, placeholder_token to ids
# token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
# # Check if initializer_token is a single token or a sequence of tokens
# if len(token_ids) > 1:
#     raise ValueError("The initializer token must be a single token.")

# initializer_token_id = token_ids[0]
# placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

# In[ ]:


# #@title Load the Stable Diffusion model
# # Load models and create wrapper for stable diffusion
# # pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
# # del pipeline
# text_encoder = CLIPTextModel.from_pretrained(
#     pretrained_model_name_or_path, subfolder="text_encoder"
# )
# vae = AutoencoderKL.from_pretrained(
#     pretrained_model_name_or_path, subfolder="vae"
# )
# unet = UNet2DConditionModel.from_pretrained(
#     pretrained_model_name_or_path, subfolder="unet"
# )

# We have added the `placeholder_token` in the `tokenizer` so we resize the token embeddings here, this will a new embedding vector in the token embeddings for our `placeholder_token`

# In[ ]:


# text_encoder.resize_token_embeddings(len(tokenizer))

#  Initialise the newly added placeholder token with the embeddings of the initializer token

# In[ ]:


# token_embeds = text_encoder.get_input_embeddings().weight.data
# token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

# In Textual-Inversion we only train the newly added embedding vector, so lets freeze rest of the model parameters here

# In[ ]:


# def freeze_params(params):
#     for param in params:
#         param.requires_grad = False

# # Freeze vae and unet
# freeze_params(vae.parameters())
# freeze_params(unet.parameters())
# # Freeze all parameters except for the token embeddings in text encoder
# params_to_freeze = itertools.chain(
#     text_encoder.text_model.encoder.parameters(),
#     text_encoder.text_model.final_layer_norm.parameters(),
#     text_encoder.text_model.embeddings.position_embedding.parameters(),
# )
# freeze_params(params_to_freeze)

# ### Creating our training data

# Let's create the Dataset and Dataloader

# In[ ]:


# %pip install -U pillow

# In[ ]:


# train_dataset = TextualInversionDataset(
#       data_root=save_path,
#       tokenizer=tokenizer,
#       size=vae.sample_size,
#       placeholder_token=placeholder_token,
#       repeats=100,
#       learnable_property=what_to_teach, #Option selected above between object and style
#       center_crop=False,
#       set="train",
# )

# In[ ]:


# def create_dataloader(train_batch_size=1):
#     return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

# Create noise_scheduler for training

# In[ ]:


# noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

# ### Training

# Define hyperparameters for our training
# If you are not happy with your results, you can tune the `learning_rate` and the `max_train_steps`

# In[ ]:


# #@title Setting up all training args
# hyperparameters = {
#     "learning_rate": 5e-04,
#     "scale_lr": True,
#     "max_train_steps": 2000,
#     "save_steps": 250,
#     "train_batch_size": 4,
#     "gradient_accumulation_steps": 1,
#     "gradient_checkpointing": True,
#     "mixed_precision": "fp16",
#     "seed": 42,
#     "output_dir": "sd-concept-output"
# }
# !mkdir -p sd-concept-output

# Train!

# In[ ]:


# #@title Training function
# logger = get_logger(__name__)

# def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
#     logger.info("Saving embeddings")
#     learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
#     learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
#     torch.save(learned_embeds_dict, save_path)

# def training_function(text_encoder, vae, unet):
#     train_batch_size = hyperparameters["train_batch_size"]
#     gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
#     learning_rate = hyperparameters["learning_rate"]
#     max_train_steps = hyperparameters["max_train_steps"]
#     output_dir = hyperparameters["output_dir"]
#     gradient_checkpointing = hyperparameters["gradient_checkpointing"]

#     accelerator = Accelerator(
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         mixed_precision=hyperparameters["mixed_precision"]
#     )

#     if gradient_checkpointing:
#         text_encoder.gradient_checkpointing_enable()
#         unet.enable_gradient_checkpointing()

#     train_dataloader = create_dataloader(train_batch_size)

#     if hyperparameters["scale_lr"]:
#         learning_rate = (
#             learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
#         )

#     # Initialize the optimizer
#     optimizer = torch.optim.AdamW(
#         text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
#         lr=learning_rate,
#     )

#     text_encoder, optimizer, train_dataloader = accelerator.prepare(
#         text_encoder, optimizer, train_dataloader
#     )

#     weight_dtype = torch.float32
#     if accelerator.mixed_precision == "fp16":
#         weight_dtype = torch.float16
#     elif accelerator.mixed_precision == "bf16":
#         weight_dtype = torch.bfloat16

#     # Move vae and unet to device
#     vae.to(accelerator.device, dtype=weight_dtype)
#     unet.to(accelerator.device, dtype=weight_dtype)

#     # Keep vae in eval mode as we don't train it
#     vae.eval()
#     # Keep unet in train mode to enable gradient checkpointing
#     unet.train()

    
#     # We need to recalculate our total training steps as the size of the training dataloader may have changed.
#     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
#     num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

#     # Train!
#     total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

#     logger.info("***** Running training *****")
#     logger.info(f"  Num examples = {len(train_dataset)}")
#     logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
#     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
#     logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
#     logger.info(f"  Total optimization steps = {max_train_steps}")
#     # Only show the progress bar once on each machine.
#     progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
#     progress_bar.set_description("Steps")
#     global_step = 0

#     for epoch in range(num_train_epochs):
#         text_encoder.train()
#         for step, batch in enumerate(train_dataloader):
#             with accelerator.accumulate(text_encoder):
#                 # Convert images to latent space
#                 latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
#                 latents = latents * 0.18215

#                 # Sample noise that we'll add to the latents
#                 noise = torch.randn_like(latents)
#                 bsz = latents.shape[0]
#                 # Sample a random timestep for each image
#                 timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

#                 # Add noise to the latents according to the noise magnitude at each timestep
#                 # (this is the forward diffusion process)
#                 noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

#                 # Get the text embedding for conditioning
#                 encoder_hidden_states = text_encoder(batch["input_ids"])[0]

#                 # Predict the noise residual
#                 noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample

#                  # Get the target for loss depending on the prediction type
#                 if noise_scheduler.config.prediction_type == "epsilon":
#                     target = noise
#                 elif noise_scheduler.config.prediction_type == "v_prediction":
#                     target = noise_scheduler.get_velocity(latents, noise, timesteps)
#                 else:
#                     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

#                 loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
#                 accelerator.backward(loss)

#                 # Zero out the gradients for all token embeddings except the newly added
#                 # embeddings for the concept, as we only want to optimize the concept embeddings
#                 if accelerator.num_processes > 1:
#                     grads = text_encoder.module.get_input_embeddings().weight.grad
#                 else:
#                     grads = text_encoder.get_input_embeddings().weight.grad
#                 # Get the index for tokens that we want to zero the grads for
#                 index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
#                 grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

#                 optimizer.step()
#                 optimizer.zero_grad()

#             # Checks if the accelerator has performed an optimization step behind the scenes
#             if accelerator.sync_gradients:
#                 progress_bar.update(1)
#                 global_step += 1
#                 if global_step % hyperparameters["save_steps"] == 0:
#                     save_path = os.path.join(output_dir, f"learned_embeds-step-{global_step}.bin")
#                     save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

#             logs = {"loss": loss.detach().item()}
#             progress_bar.set_postfix(**logs)

#             if global_step >= max_train_steps:
#                 break

#         accelerator.wait_for_everyone()


#     # Create the pipeline using using the trained modules and save it.
#     if accelerator.is_main_process:
#         pipeline = StableDiffusionPipeline.from_pretrained(
#             pretrained_model_name_or_path,
#             text_encoder=accelerator.unwrap_model(text_encoder),
#             tokenizer=tokenizer,
#             vae=vae,
#             unet=unet,
#         )
#         pipeline.save_pretrained(output_dir)
#         # Also save the newly trained embeddings
#         save_path = os.path.join(output_dir, f"learned_embeds.bin")
#         save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

# In[ ]:


# import accelerate
# accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet))

# for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
#   if param.grad is not None:
#     del param.grad  # free some memory
#   torch.cuda.empty_cache()

# In[ ]:


# #@title Save your newly created concept to the [library of concepts](https://huggingface.co/sd-concepts-library)?

# save_concept_to_public_library = True #@param {type:"boolean"}
# name_of_your_concept = "Cat toy" #@param {type:"string"}
# #@markdown `hf_token_write`: leave blank if you logged in with a token with `write access` in the [Initial Setup](#scrollTo=KbzZ9xe6dWwf). If not, [go to your tokens settings and create a write access token](https://huggingface.co/settings/tokens)
# hf_token_write = "" #@param {type:"string"}

# if(save_concept_to_public_library):
#   from slugify import slugify
#   from huggingface_hub import HfApi, HfFolder, CommitOperationAdd
#   from huggingface_hub import create_repo
#   repo_id = f"sd-concepts-library/{slugify(name_of_your_concept)}"
#   output_dir = hyperparameters["output_dir"]
#   if(not hf_token_write):
#     with open(HfFolder.path_token, 'r') as fin: hf_token = fin.read();
#   else:
#     hf_token = hf_token_write
#   #Join the Concepts Library organization if you aren't part of it already
#   !curl -X POST -H 'Authorization: Bearer '$hf_token -H 'Content-Type: application/json' https://huggingface.co/organizations/sd-concepts-library/share/VcLXJtzwwxnHYCkNMLpSJCdnNFZHQwWywv
#   images_upload = os.listdir("my_concept")
#   image_string = ""
#   repo_id = f"sd-concepts-library/{slugify(name_of_your_concept)}"
#   for i, image in enumerate(images_upload):
#       image_string = f'''{image_string}![{placeholder_token} {i}](https://huggingface.co/{repo_id}/resolve/main/concept_images/{image})
# '''
#   if(what_to_teach == "style"):
#       what_to_teach_article = f"a `{what_to_teach}`"
#   else:
#       what_to_teach_article = f"an `{what_to_teach}`"
#   readme_text = f'''---
# license: mit
# base_model: {pretrained_model_name_or_path}
# ---
# ### {name_of_your_concept} on Stable Diffusion
# This is the `{placeholder_token}` concept taught to Stable Diffusion via Textual Inversion. You can load this concept into the [Stable Conceptualizer](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_conceptualizer_inference.ipynb) notebook. You can also train your own concepts and load them into the concept libraries using [this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb).

# Here is the new concept you will be able to use as {what_to_teach_article}:
# {image_string}
# '''
#   #Save the readme to a file
#   readme_file = open("README.md", "w")
#   readme_file.write(readme_text)
#   readme_file.close()
#   #Save the token identifier to a file
#   text_file = open("token_identifier.txt", "w")
#   text_file.write(placeholder_token)
#   text_file.close()
#   #Save the type of teached thing to a file
#   type_file = open("type_of_concept.txt","w")
#   type_file.write(what_to_teach)
#   type_file.close()
#   operations = [
#     CommitOperationAdd(path_in_repo="learned_embeds.bin", path_or_fileobj=f"{output_dir}/learned_embeds.bin"),
#     CommitOperationAdd(path_in_repo="token_identifier.txt", path_or_fileobj="token_identifier.txt"),
#     CommitOperationAdd(path_in_repo="type_of_concept.txt", path_or_fileobj="type_of_concept.txt"),
#     CommitOperationAdd(path_in_repo="README.md", path_or_fileobj="README.md"),
#   ]
#   create_repo(repo_id,private=True, token=hf_token)
#   api = HfApi()
#   api.create_commit(
#     repo_id=repo_id,
#     operations=operations,
#     commit_message=f"Upload the concept {name_of_your_concept} embeds and token",
#     token=hf_token
#   )
#   api.upload_folder(
#     folder_path=save_path,
#     path_in_repo="concept_images",
#     repo_id=repo_id,
#     token=hf_token
#   )

# In[ ]:


#@title Set up the pipeline 
from diffusers import DPMSolverMultistepScheduler
pipe = StableDiffusionPipeline.from_pretrained(
    args.output_dir,
    scheduler=DPMSolverMultistepScheduler.from_pretrained(args.output_dir), subfolder="scheduler"),
    torch_dtype=torch.float16,
).to("cuda")

# In[ ]:


#@title Run the Stable Diffusion pipeline
#@markdown Don't forget to use the placeholder token in your prompt

prompt = "a \u003Ccat-toy> inside ramen-bowl" #@param {type:"string"}

num_samples = 2 #@param {type:"number"}
num_rows = 1 #@param {type:"number"}

all_images = [] 
for _ in range(num_rows):
    images = pipe([prompt] * num_samples, num_inference_steps=30, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_rows, num_samples)
grid

# In[ ]:



