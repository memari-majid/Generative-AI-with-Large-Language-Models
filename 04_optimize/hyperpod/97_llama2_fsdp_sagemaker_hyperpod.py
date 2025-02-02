#!/usr/bin/env python
# coding: utf-8

# # Get Started Training Llama 2 with PyTorch FSDP in 5 Minutes
# 
# _Based on this repo: https://github.com/aws-samples/awsome-distributed-training/_
# 
# These scripts provide an easy way to get started with multinode [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) training on Slurm. It is designed to be as simple as possible, requires no data preparation, and uses a simple Conda environment. 
# 
# ## 0. Prerequisites
# 
# Before running this training, you'll need to create a Slurm cluster with an FSx for Lustre file system. Instructions can be found in [1.architectures](../../1.architectures).
# 
# ## 1. Create Environment
# 
# On your cluster head node, 
# 1. Navigate to your shared FSx for Lustre file system.
# * If you followed the tutorial linked above, it will be location at `/fsx`.   
# 2. Clone this repo. 
# 
# ```
# cd /fsx
# git clone https://github.com/aws-samples/awsome-distributed-training/
# cd awsome-distributed-training/3.test_cases/10.FSDP
# ```
# 
# 3. Run the `0.create_conda_env.sh` script. 
# * This script will first download and install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/), then create a Conda env called `pt_fsdp`.
# 
# ```
# bash 0.create_conda_env.sh
# ```
# 
# * By creating this environment on the shared FSx for Lustre volume, all compute nodes in our cluster will have access to it.
# 
# ## 2. Data
# 
# For this example, we'll be using the [C4 dataset](https://huggingface.co/datasets/allenai/c4), which is several hundred gigabytes. Instead of downloading the whole thing, the `create_streaming_dataloaders` function will stream the dataset from [HuggingFace](https://huggingface.co/datasets), so there's no data prep required for running this training. 
# 
# If you'd like to instead use your own dataset, you can do so by [formatting it as a HuggingFace dataset](https://huggingface.co/docs/datasets/create_dataset), and passing its location to the `--dataset_path` argument.
# 
# ## 3. Launch Training
# 
# The script to launch a Slurm batch training job can be found in `1.distributed_training.sbatch`. You can adjust the number of training nodes by modifying `#SBATCH --nodes=4`. You can also adjust the training parameters in `TRAINING_ARGS`. Additional parameters can be found in `model/arguments.py`. Note that we use the same directory for both `--checkpoint_dir` and `--resume_from_checkpoint`. If there are multiple checkpoints, `--resume_from_checkpoint` will automatically select the most recent one. This way if our training is interupted for any reason, it will automatically pick up the most recent checkpoint.
# 
# ```
# declare -a TRAINING_ARGS=(
#     --num_key_value_heads=32 \
#     --llama_intermediate_size=11008 \
#     --max_context_width=4096 \
#     --hidden_width=4096 \
#     --num_layers=32 \
#     --num_heads=32 \
#     --model_type=llama_v2 \
#     --checkpoint_freq=1000 \
#     --validation_freq=500 \
#     --checkpoint_dir=./checkpoints \
#     --resume_from_checkpoint=./checkpoints
# )
# ```
# 
# To launch your training, run
# 
# ```
# sbatch 1.distributed_training.sbatch
# ```
# 
# You'll find a new file in the FSDP directory of the form `slurm-[job-number].out`. This will be continuously updated with your training logs. Don't be worried if you see a long stream of NCCL logs (we prefer to use `NCCL_DEBUG=INFO` for verbose logging). After about a minute, you should see your model training, with an output similar to below.
# 
# ```
# + TORCHRUN=./pt_fsdp/bin/torchrun
# + export TRAIN_SCRIPT=./train.py
# + TRAIN_SCRIPT=./train.py
# + TRAINING_ARGS=(--max_context_width=4096 --num_key_value_heads=32 \ # 7b: 32 13b: 40 70b: 8 --llama_intermediate_size=11008 \ # 7b: 11008 13b: 13824 70b: 28672 --hidden_width=4096 \ # 7b: 4096 13b: 5120 70b: 8192 --num_layers=32 \ # 7b: 32 13b: 40 70b: 80 --num_heads=32 \ # 7b: 32 13b: 40 70b: 64 --model_type=llama_v2 --checkpoint_freq=50 --validation_freq=500 --checkpoint_dir=./checkpoints --resume_from_checkpoint=./checkpoints)
# ...
# 0: 2023-11-29 04:17:52 I [train.py:175] Creating Model
# 0: 2023-11-29 04:19:17 I [train.py:182] Created model with total parameters: 6889410560 (6.89 B)
# 0: 2023-11-29 04:19:28 I [train.py:209] Wrapped model with FSDP
# 0: 2023-11-29 04:19:28 I [train.py:226] Created optimizer
# ...
# 2: ip-10-1-41-139:6171:8092 [0] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.7.3-aws
# 3: ip-10-1-44-54:6168:6168 [7] NCCL INFO cudaDriverVersion 12020
# 0: ip-10-1-14-81:6158:9214 [2] NCCL INFO NET/OFI Selected Provider is efa (found 4 nics)
# ...
# 0: ip-10-1-14-81:6158:9214 [2] NCCL INFO comm 0x8b6b550 rank 2 nranks 32 cudaDev 2 busId 201c0 - Init COMPLETE
# 0: ip-10-1-14-81:6157:9213 [1] NCCL INFO comm 0x8494480 rank 1 nranks 32 cudaDev 1 busId 101d0 - Init COMPLETE
# 0: 2023-11-29 04:19:48 I [train.py:122] Batch 0 Loss: 11.6533041, Speed: 3.98 samples/sec, lr: 0.000006
# 0: 2023-11-29 04:19:54 I [train.py:122] Batch 1 Loss: 11.620493, Speed: 10.72 samples/sec, lr: 0.000013
# 0: 2023-11-29 04:20:00 I [train.py:122] Batch 2 Loss: 11.3152923, Speed: 10.71 samples/sec, lr: 0.000019
# 0: 2023-11-29 04:20:06 I [train.py:122] Batch 3 Loss: 10.461415, Speed: 10.11 samples/sec, lr: 0.000025
# 0: 2023-11-29 04:20:12 I [train.py:122] Batch 4 Loss: 11.8934202, Speed: 10.71 samples/sec, lr: 0.000031
# 0: 2023-11-29 04:20:18 I [train.py:122] Batch 5 Loss: 13.9545879, Speed: 10.70 samples/sec, lr: 0.000038
# ```
# 
# To modify training for a 13 or 70B Llama 2 model, just change the corresponding parameters based on the values in the [Llama 2 paper](https://arxiv.org/abs/2307.09288).
# 
# | Param                    |     7B      |     13B     |     70B     |
# | ------------------------ | ----------- | ----------- | ----------- |
# | llama_intermediate_size  | 11008       | 13824       | 28672       |
# | num_key_value_heads      | 32          | 40          | 8           |
# | hidden_width             | 4096        | 5120        | 8192        |
# | num_layers               | 32          | 40          | 80          |
# | num_heads                | 32          | 40          | 64          |
# 
# If you need to cancel or modify your job, see the Slurm commands available in the [Slurm documentation](https://slurm.schedmd.com/quickstart.html).

# ## Overview
# 
# This guide assumes that you have the following:
# * A functional Slurm cluster on AWS. We also assume that Ubuntu AMI is used.
# * Neuron SDK is installed on the cluster (see [AWS Neuron SDK documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx) for the steps).
# * An FSx for Lustre filesystem mounted on `/fsx`.

# ## 1. Setup Environment
# First of all, you need to have a Python virtual environment for `torch-neuronx` under `APPS_PATH`.
# 
# ```bash
# bash 1.setup-venv.sh ${APPS_PATH} # The argument specifies APPS_PATH
# ```

# ## 2. `neuronx-nemo-megatron` library need to be installed (and initialized) in the environment.
# 
# 
# ```bash
# bash 2.setup-neuronx-nemo-megatron.sh ${APPS_PATH} #
# ```
# You will see the following ERROR line during the script execution. This is safe to ignore.
# 
# ```console
# + python3 -c 'from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import compile_helper; compile_helper()'
# 2023-Nov-18 09:17:45.728072 175272:175272 ERROR  TDRV:tdrv_get_dev_info                       No neuron device available
# ```

# ## 1. Prepare Llama2 model
# 
# We recommend that you setup a Slurm cluster using the template in the architectures directory. Before creating the Slurm cluster, you need to setup the following environment variables:
# 
# ```bash
# export APPS_PATH=/fsx
# export FSX_PATH=/fsx
# export MODEL_PATH=/fsx
# export DATA_PATH=$FSX_PATH/data/books
# export TEST_CASE_PATH=${APPS_PATH}/awsome-distributed-training/3.test_cases/8.neuronx-nemo-megatron  # where you copy the test case or set to your test case path
# ```
# 
# This test case requires Llama2 model, which governed by the Meta license and must be downloaded and converted to the standard [Hugging Face](https://huggingface.co/) format prior to running this sample.
# You can submit access request from [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/), we need "Llama 2 & Llama Chat" to be checked. Use the [download.sh](https://github.com/facebookresearch/llama/blob/main/download.sh) in the official repository. You will be asked to input an URL from the email you recieve from meta.  
# 
# We will assume that you had placed the model and tokenizer as follows on cluster:
# 
# ```
# ${MODEL_PATH}/Llama2-meta/
# ├── 7B/
# │   ├── checklist.chk
# │   ├── consolidated.00.pth
# │   └── params.json
# ├── tokenizer.model
# └── tokenizer_checklist.chk
# ```
# 

# ## 3. Convert weights to HuggingFace format
# To convert the model to the standard Hugging Face format, the following script in transformers can be called with the following (example) command:
# 
# ```
# sbatch 3.convert-weight.sbatch
# ```

# You can check progress of with `tail` command.
# 
# ```
# $ tail -f slurm-3.convert-weight.sbatch-xxx.out 
# ```
# 
# ```console
# Fetching all parameters from the checkpoint at /fsx/Llama2-meta/7B.
# Loading the checkpoint in a Llama model.
# Loading checkpoint shards: 100%|██████████| 33/33 [00:12<00:00,  2.65it/s]
# ...
# ```
# 
# Once the job completed, you will have the Llama-2-7b model weights and tokenizer in a huggingface format under a directory called `Llama2-7b-hf` with the following format:
# 
# ```console
# ${DATAPATH}/Llama2-7b-hf/
# ├── config.json
# ├── generation_config.json
# ├── pytorch_model-00001-of-00002.bin
# ├── pytorch_model-00002-of-00002.bin
# ├── pytorch_model.bin.index.json
# ├── special_tokens_map.json
# ├── tokenizer.json
# ├── tokenizer.model
# └── tokenizer_config.json
# ```

# ## 4. Download and Tokenize dataset
# This tutorial makes use of a [Red pyjama dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T). The dataset can be downloaded to your cluster by running the following commands on the head node:
# 
# ```bash
# mkdir -p ${DATA_PATH} 
# wget https://data.together.xyz/redpajama-data-1T/v1.0.0/book/book.jsonl -O ${DATA_PATH}/book.jsonl # Note: Dataset download is 50G and will take approximately 3-4 hours to download. You can also use https://aria2.github.io/ for faster download
# # or
# # wget https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/book_sample.jsonl -O ${DATA_PATH}/book.jsonl # Smaller sample dataset for quick testing
# ```
# 
# Once you have the Tokenizer and the dataset. You can tokenize the dataset following the below command: 
# ```bash
# sbatch 4.tokenize.sbatch
# ```

# Post tokenizing the dataset, you will have a path to the tokenizer and the dataset which will be used for pretraining. 
# 
# ## Llama2 training configurations
# We tested with the following model sizes: 7B
# ### Llama2 7B
# 
# - Model configuration
#     - Attention heads: 32
#     - Layers: 32
#     - Sequence length: 4096
#     - Hidden size: 4096
#     - Hidden FFN size: 11008
#     - Microbatch size: 1
#     - Global batch size: 256
# 
# - Distributed training configuration
#     - Number of nodes: 4
#     - Tensor parallel degree: 8
#     - Pipeline parallel degree: 1
#     - Data parallel degree: 16
# 

# ## 5. Pre-compile the model
# By default, PyTorch Neuron uses a just in time (JIT) compilation flow that sequentially compiles all of the neural network compute graphs as they are encountered during a training job. The compiled graphs are cached in a local compiler cache so that subsequent training jobs can leverage the compiled graphs and avoid compilation (so long as the graph signatures and Neuron version have not changed).
# 
# An alternative to the JIT flow is to use the included [neuron_parallel_compile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile.html?highlight=neuron_parallel_compile) command to perform ahead of time (AOT) compilation. In the AOT compilation flow, the compute graphs are first identified and extracted during a short simulated training run, and the extracted graphs are then compiled and cached using parallel compilation, which is considerably faster than the JIT flow.
# 
# Before starting the compilation you need to update your path to the dataset and tokenizer in the llama_7b script as below : 
# 
# ```bash
# cd ${APPS_PATH}/neuronx-nemo-megatron/nemo/examples/nlp/language_modeling
# vi test_llama.sh
# ```
# 
# Update the below lines
# 
# ```bash
# : ${TOKENIZER_PATH=$HOME/llamav2_weights/7b-hf}
# : ${DATASET_PATH=$HOME/examples_datasets/llama_7b/book.jsonl-processed_text_document}
# ```
# 
# to
# ```bash
# : ${TOKENIZER_PATH=${MODEL_PATH}/Llama2-7b-hf}
# : ${DATASET_PATH=${DATA_PATH}/book-tokenized_text_document}
# ```
# 

# Then, run the following command to launch an AOT pre-compilation job on your ParallelCluster:
# 
# ```bash
# bash 5.precompile-model.sh
# ```
# 
# Once you have launched the precompilation job, run the `squeue` command to view the SLURM job queue on your cluster. If you have not recently run a job on your cluster, it may take 4-5 minutes for the requested trn1.32xlarge nodes to be launched and initialized. Once the job is running, `squeue` should show output similar to the following:
# 
# ```console
#     JOBID  PARTITION  NAME           USER    ST  TIME  NODES NODELIST(REASON)
#     10     compute1   compile.slurm  ubuntu  R   5:11  4     compute1-dy-queue1-i1-[1-4]
# ```
# 
# You can view the output of the precompilation job by examining the file named `slurm-compile.slurm-ZZ.out` where ZZ represents the JOBID of your job in the `squeue` output, above. Ex:
# ```
# tail -f slurm-compile.slurm-10.out
# ```
# 
# Once the precompilation job is complete, you should see a message similar to the following in the logs:
# 
# ```console
# 2023-06-11 23:04:08.000738: INFO ||PARALLEL_COMPILE||: Total graphs: 22
# 2023-06-11 23:04:08.000738: INFO ||PARALLEL_COMPILE||: Total successful compilations: 22
# 2023-06-11 23:04:08.000738: INFO ||PARALLEL_COMPILE||: Total failed compilations: 0
# ```
# 
# At this point, you can press `CTRL-C` to exit the tail command.
# 
# 

# ## 6. Launch a pretraining job
# 
# Submit the training job
# 
# ```
# bash 6.pretrain-model.sh
# ```
# 
# 
# As outlined above, you can again use the `squeue` command to view the job queue. Once you see that your pretraining job is running, you can view the output of the training job by examining the file named `slurm-run.slurm-ZZ.out` where ZZ represents the JOBID of your job:
# 
# ```bash
# tail -f slurm-run.slurm-11.out
# ```
# 
# Once the model is loaded onto the Trainium accelerators and training has commenced, you will begin to see output indicating the job progress:
# 
# ```console
# Epoch 0:  22%|██▏       | 4499/20101 [22:26:14<77:48:37, 17.95s/it, loss=2.43, v_num=5563, reduced_train_loss=2.470, gradient_norm=0.121, parameter_norm=1864.0, global_step=4512.0, consumed_samples=1.16e+6, iteration_time=16.40]
# Epoch 0:  22%|██▏       | 4500/20101 [22:26:32<77:48:18, 17.95s/it, loss=2.43, v_num=5563, reduced_train_loss=2.470, gradient_norm=0.121, parameter_norm=1864.0, global_step=4512.0, consumed_samples=1.16e+6, iteration_time=16.40]
# Epoch 0:  22%|██▏       | 4500/20101 [22:26:32<77:48:18, 17.95s/it, loss=2.44, v_num=5563, reduced_train_loss=2.450, gradient_norm=0.120, parameter_norm=1864.0, global_step=4512.0, consumed_samples=1.16e+6, iteration_time=16.50]
# ```
# 

# ## Contributors
# 
# * [A] Keita Watanabe - mlkeita@
# * [R] Verdi March - marcverd@
# * [R] Brad Doran 
# * [R] Justin Pirtle 
# * [R] Pierre-Yves Aquilanti - pierreya@
# 
