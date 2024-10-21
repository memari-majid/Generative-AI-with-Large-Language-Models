# Generative AI with LLMs on AWS

This repository contains code examples originally created by the authors of the [Generative AI on AWS](https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/) textbook and updated by Majid Memari. The repository also includes a Streamlit web application that builds a knowledge base from course content, allowing students to interactively ask questions and receive answers using the Code Llama language model. Additionally, a virtual TA is integrated into the system, providing answers based on course materials and sample codes.

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Notes](#notes)
- [License](#license)

## Features
- Automatically reads and processes all files in the current directory and subdirectories.
- Supports a variety of file types that can be read as text.
- Builds a knowledge base using embeddings generated by Hugging Face models.
- Uses Code Llama for language understanding and question answering.
- Provides a web-based user interface using Streamlit.
- Displays source documents related to the answers for better context.

## How It Works
Upon execution, the application follows these steps:

1. **File Loading**: Scans the current directory and recursively reads all files as text.
2. **Document Splitting**: Breaks down the text into smaller chunks for efficient processing.
3. **Embedding Generation**: Generates embeddings for each text chunk using Hugging Face models.
4. **Vector Store Creation**: Stores embeddings in a FAISS vector store for efficient similarity searches.
5. **Model Initialization**: Loads the Code Llama model for language understanding and question answering.
6. **User Interaction**: Provides a web interface where users can input questions.
7. **Answer Generation**: Retrieves relevant documents and generates answers using the Code Llama model.

## Requirements
- Python 3.9 or higher
- Hardware requirements:
  - For the 7B Code Llama model: At least 13GB of GPU VRAM or sufficient CPU RAM.
  - Larger models will require additional VRAM or RAM.
- Required libraries:
  - `streamlit`
  - `langchain`
  - `transformers`
  - `torch`
  - `faiss-cpu` or `faiss-gpu`
  - `sentence-transformers`

## Installation
To set up the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/memari-majid/generative-ai-on-aws.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd generative-ai-on-aws
   ```

3. **Create a Virtual Environment (Optional but Recommended)**:
   - For virtualenv:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - For conda:
     ```bash
     conda create -n knowledge_base_env python=3.9
     conda activate knowledge_base_env
     ```

4. **Install Required Libraries**:
   ```bash
   pip install streamlit langchain transformers torch faiss-cpu sentence-transformers
   ```

   If you have a GPU with CUDA support, install PyTorch and FAISS with CUDA:

   ```bash
   pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
   pip install faiss-gpu
   ```

5. **Ensure CUDA Drivers Are Installed (If Using GPU)**:
   Ensure that your system has the correct CUDA drivers installed.

6. **Download the Code Llama Model**:
   The model will be automatically downloaded when the application is first run. Ensure you have sufficient disk space and a stable internet connection.

## Usage
To run the application:

```bash
streamlit run main.py
```

Open the URL provided by Streamlit (usually [http://localhost:8501](http://localhost:8501)) in your browser.

The application will:

- Load all readable files from the current directory.
- Build the knowledge base.
- Initialize the Code Llama model.
- You can then enter questions in the input box and receive answers based on the course content.

## Technical Details

- **File Loading and Preprocessing**: Python's `os.walk` is used to traverse directories and read files as text. Non-readable files are skipped, with appropriate warnings logged.
- **Document Splitting**: LangChain's `RecursiveCharacterTextSplitter` splits documents into chunks of 1000 characters with a 100-character overlap.
- **Embedding Generation**: Embeddings are created using Hugging's `sentence-transformers/all-MiniLM-L6-v2` model via `HuggingFaceEmbeddings`.
- **Vector Store**: The generated embeddings are stored in a FAISS vector store, allowing efficient similarity search for document retrieval.
- **Code Llama Integration**: Code Llama is loaded using Hugging Face Transformers and integrated into the QA pipeline with LangChain'ss `HuggingFacePipeline`.
- **Question Answering**: The system retrieves relevant documents from the vector store based on similarity and generates answers using the Code Llama model.

## Notes
- **Model Size**: The default model is Code Llama 7B. To use a different model, modify the `model_name` variable in the script.
- **Hardware Requirements**: Larger models need more VRAM or RAM, so ensure your system has adequate resources.
- **Performance**: Using a GPU can significantly speed up performance. CPU inference may be slower for larger models.
- **Licensing**: Ensure compliance with the licensing terms of Code Llama when using the model.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
