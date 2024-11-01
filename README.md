# Generative AI with Large Language Models

This repository is an enhanced version of the code examples from the O'Reilly textbook [Generative AI on AWS](https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/) by Chris Fregly, Antje Barth, and Shelbee Eigenbrode (November 2023, ISBN: 9781098159221). The examples have been updated and optimized to work with Cursor AI, providing an interactive learning experience for working with Large Language Models (LLMs) on AWS.

> **Note**: This repository is regularly updated to incorporate state-of-the-art AI developments and best practices, extending beyond the book's original content.

## Table of Contents
- [Generative AI with Large Language Models](#generative-ai-with-large-language-models)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Repository Structure](#repository-structure)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Using with Cursor AI](#using-with-cursor-ai)
  - [Technical Details](#technical-details)
    - [AWS Services Used](#aws-services-used)
    - [Key Features](#key-features)
  - [License](#license)

## Overview

This repository serves multiple purposes:
- Provides updated code examples from the "Generative AI on AWS" textbook
- Integrates Cursor AI for enhanced code understanding and development
- Incorporates latest developments in AI and AWS services
- Offers practical implementations of advanced LLM concepts

## Repository Structure

The repository follows the book's chapter organization with additional enhancements:

- `01_intro/`: Chapter 1 - Generative AI Concepts
- `02_prompt/`: Chapter 2 - Prompt Engineering and In-Context Learning
- `03_foundation/`: Chapter 3 - Large-Language Foundation Models
- `04_optimize/`: Chapter 4 - Memory and Compute Optimizations
- `05_finetune/`: Chapter 5 - Instruction Fine-Tuning
- `06_peft/`: Chapter 6 - Parameter-Efficient Fine-Tuning (PEFT)
- `07_rlhf/`: Chapter 7 - Reinforcement Learning from Human Feedback (RLHF)
- `08_deploy/`: Chapter 8 - Model Deployment Optimizations
- `09_rag/`: Chapter 9 - Context-Aware Reasoning with RAG and Agents
- `10_multimodal/`: Chapter 10 - Exploring Multimodal Foundation Models
- `11_diffusers/`: Chapter 11 - Controlled Image Generation and Fine-Tuning with Stable Diffusion
- `12_bedrock/`: Chapter 12 - Exploring Amazon Bedrock and Fine-Tuning Foundation Models

Each directory contains practical implementations and examples that are regularly updated with the latest AI developments.

## Requirements

- Python 3.9 or higher
- AWS account with appropriate permissions
- GPU recommended for training and inference (16GB+ VRAM for some examples)
- Required libraries listed in requirements.txt

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/memari-majid/generative-ai-on-aws.git
   cd generative-ai-on-aws
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   
   # For GPU support (optional but recommended)
   pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
   ```

4. **Configure AWS Credentials**:
   ```bash
   aws configure
   ```
   Enter your AWS Access Key ID, Secret Access Key, default region, and output format.

## Using with Cursor AI

[Cursor](https://cursor.sh/) is an AI-powered code editor that can help you understand and work with this codebase more effectively:

1. **Install Cursor**:
   - Download and install Cursor from https://cursor.sh/

2. **Open the Project**:
   - Launch Cursor
   - File -> Open Folder -> Select the generative-ai-on-aws directory

3. **Using AI Features**:
   - Press `Cmd/Ctrl + K` to chat with the AI about any code
   - Press `Cmd/Ctrl + L` to get AI code suggestions
   - Highlight code and press `Cmd/Ctrl + Shift + I` for explanations
   - Use `/edit` command in chat to get code modifications

4. **Best Practices**:
   - Ask about specific files or functions for targeted help
   - Request explanations of complex code sections
   - Use AI to understand AWS service integrations
   - Get help with debugging and error messages

## Technical Details

### AWS Services Used
- Amazon SageMaker for training and deployment
- Amazon S3 for storage
- AWS Lambda for serverless functions
- Amazon Bedrock for foundation models
- Amazon OpenSearch for vector search

### Key Features
- Fine-tuning large language models
- Implementing PEFT techniques
- Deploying models to production
- Building RAG applications
- Working with multimodal models
- Using diffusion models
- Implementing RLHF

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---
Based on the book "Generative AI on AWS" by Chris Fregly, Antje Barth, and Shelbee Eigenbrode (O'Reilly Media, Inc., November 2023, ISBN: 9781098159221). Repository maintained and enhanced by Majid Memari.
