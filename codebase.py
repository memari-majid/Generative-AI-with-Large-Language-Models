# main.py

import os
import sys
import logging
import streamlit as st
from typing import List, Iterator
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Database connection parameters (Replace with your actual credentials)
DB_NAME = 'knowledge_base_db'
DB_USER = 'postgres'
DB_PASSWORD = '1234'
DB_HOST = 'localhost'
DB_PORT = '5432'
COLLECTION_NAME = 'python_documents_collection'

# Define MAX_NEW_TOKENS as a constant
MAX_NEW_TOKENS = 256  # Adjust as needed

def setup_database():
    """
    Connects to PostgreSQL, creates the database and pgvector extension if they don't exist.
    """
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Create database if it doesn't exist
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_NAME,))
        exists = cur.fetchone()
        if not exists:
            cur.execute(f'CREATE DATABASE {DB_NAME}')
            logging.info(f"Database '{DB_NAME}' created.")
        else:
            logging.info(f"Database '{DB_NAME}' already exists.")

        cur.close()
        conn.close()

        # Connect to the new database to create the extension
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Create pgvector extension if it doesn't exist
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        logging.info("PGVector extension ensured.")

        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Error setting up database: {e}")
        st.error(f"Error setting up database: {e}")
        sys.exit(1)

def list_chapters(directory_path: str) -> List[str]:
    """
    Scans the given directory and returns a sorted list of chapter names.
    Chapters are identified by directory names starting with digits followed by an underscore.
    Example: '01_intro', '02_prompt', etc.
    """
    chapters = []
    pattern = re.compile(r'^\d+_.+')
    try:
        for entry in os.scandir(directory_path):
            if entry.is_dir() and pattern.match(entry.name):
                chapters.append(entry.name)
        chapters_sorted = sorted(chapters)
        logging.info(f"Chapters found: {chapters_sorted}")
        return chapters_sorted
    except Exception as e:
        logging.error(f"Error listing chapters: {e}")
        st.error(f"Error listing chapters: {e}")
        return []

def load_files_in_batches(directory_path: str, batch_size: int) -> Iterator[List[Document]]:
    """
    Generator that yields batches of Documents from the specified directory.
    Only processes .py and .ipynb files, extracts Python code, and includes chapter metadata.
    Also processes README.md files to extract key concepts.
    """
    allowed_extensions = ['.py', '.ipynb']
    readme_filename = 'README.md'

    documents = []
    count = 0
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            _, ext = os.path.splitext(file_name)
            if ext.lower() not in allowed_extensions and file_name.lower() != readme_filename.lower():
                logging.info(f"Skipping file '{file_name}' with unsupported extension '{ext}'")
                continue
            file_path = os.path.join(root, file_name)
            try:
                # Extract chapter name from the file path
                # Assuming the directory structure is ./chapter_name/file
                relative_path = os.path.relpath(file_path, directory_path)
                parts = relative_path.split(os.sep)
                if len(parts) < 2:
                    logging.warning(f"File '{file_path}' is not within a chapter directory. Skipping.")
                    continue
                chapter = parts[0]

                if not re.match(r'^\d+_.+', chapter):
                    logging.info(f"Skipping file '{file_path}' as it is not within a valid chapter directory.")
                    continue

                if file_name.lower() == readme_filename.lower():
                    # Handle README.md
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    # Remove NUL characters
                    content = content.replace('\x00', '')
                    # Skip empty content
                    if not content.strip():
                        logging.warning(f"README '{file_path}' is empty after removing NUL characters. Skipping.")
                        continue
                    doc = Document(page_content=content, metadata={"source": file_path, "chapter": chapter, "type": "README"})
                    documents.append(doc)
                    count += 1
                    if count % batch_size == 0:
                        yield documents
                        documents = []
                    continue

                if ext.lower() == '.ipynb':
                    # Handle Jupyter notebooks
                    import nbformat
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        nb = nbformat.read(f, as_version=4)
                    content = ''
                    for cell in nb.cells:
                        if cell.cell_type == 'code':
                            # Optionally, check if the code is Python
                            if cell.metadata.get('language', 'python').lower() == 'python':
                                content += cell.source + '\n'
                else:
                    # Handle .py files
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                # Remove NUL characters
                content = content.replace('\x00', '')
                # Skip empty content
                if not content.strip():
                    logging.warning(f"File '{file_path}' is empty after removing NUL characters. Skipping.")
                    continue
                doc = Document(page_content=content, metadata={"source": file_path, "chapter": chapter, "type": "code"})
                documents.append(doc)
                count += 1
                if count % batch_size == 0:
                    yield documents
                    documents = []
            except Exception as e:
                logging.warning(f"Could not read '{file_path}'. Skipping. Error: {e}")
    if documents:
        yield documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller chunks for better processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def process_and_store_embeddings_in_batches(vectorstore: PGVector, documents: Iterator[List[Document]]):
    """
    Processes documents and stores their embeddings in the vector store in batches.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    for batch_docs in documents:
        split_docs = split_documents(batch_docs)
        try:
            vectorstore.add_documents(split_docs)
            logging.info(f"Processed and stored a batch of {len(split_docs)} documents.")
        except Exception as e:
            logging.error(f"Error storing embeddings in vector store: {e}")
            st.error(f"Error storing embeddings in vector store: {e}")

def truncate_input(input_text: str, tokenizer: AutoTokenizer, max_input_length: int) -> str:
    """
    Truncates the input text to ensure it doesn't exceed the maximum input length.
    """
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    if len(input_ids) > max_input_length:
        input_ids = input_ids[:max_input_length]
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return input_text

def main():
    st.set_page_config(page_title="Python Knowledge Base", layout="wide")

    st.title("Python Knowledge Base")
    st.write("""
    This application automatically reads Python code from `.py` files and Jupyter notebooks (`.ipynb`) in the current directory, organizes them by chapters, and creates a knowledge base.
    Each chapter contains a `README.md` file that outlines the chapter's concepts and objectives.
    Select a chapter and ask questions about the Python code and its applications within that chapter.
    """)

    # Database setup
    setup_database()
    connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # List available chapters
    chapters = list_chapters('.')
    if not chapters:
        st.error("No chapters found. Ensure that chapter directories are named like '01_intro', '02_prompt', etc.")
        return

    # Chapter selection
    selected_chapter = st.selectbox("Select a Chapter", options=chapters)

    # Initialize session state for vectorstore
    if 'vectorstore' not in st.session_state:
        with st.spinner("Processing Python files and building the vector store..."):
            try:
                # Create a new PGVector vectorstore
                vectorstore = PGVector(
                    connection_string=connection_string,
                    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                    collection_name=COLLECTION_NAME,
                    pre_delete_collection=True  # Deletes existing collection before creating a new one
                )
                # Process files in batches and store embeddings
                batch_size = 10  # Adjust based on memory constraints
                document_batches = load_files_in_batches('.', batch_size)
                process_and_store_embeddings_in_batches(vectorstore, document_batches)
                st.session_state['vectorstore'] = vectorstore
                st.success("Vector store created and loaded into session.")
            except Exception as e:
                st.error(f"Failed to create vector store: {e}")
                logging.error(f"Failed to create vector store: {e}")
                return
    else:
        vectorstore = st.session_state['vectorstore']
        logging.info("Vector store loaded from session state.")

    # Initialize Code Llama LLM and Tokenizer
    if 'llm' not in st.session_state or 'tokenizer' not in st.session_state:
        try:
            with st.spinner("Loading Code Llama model..."):
                model_name = "codellama/CodeLlama-13b-hf"  # You can choose other variants like 13b or 34b
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                # Create a pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=MAX_NEW_TOKENS,  # Use max_new_tokens instead of max_length
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.15
                )
                llm = HuggingFacePipeline(pipeline=pipe)
                # Store both llm and tokenizer in session state
                st.session_state['llm'] = llm
                st.session_state['tokenizer'] = tokenizer
                st.session_state['pipe'] = pipe  # Optional: Store the pipeline if needed elsewhere
                logging.info("LLM and tokenizer initialized and loaded into session state.")
        except Exception as e:
            st.error(f"Failed to initialize the Code Llama model: {e}")
            logging.error(f"Failed to initialize the Code Llama model: {e}")
            return
    else:
        llm = st.session_state['llm']
        tokenizer = st.session_state['tokenizer']
        pipe = st.session_state.get('pipe')  # Optional

    # Create a retriever with a filter for the selected chapter
    try:
        # Sanitize the selected chapter to prevent SQL injection
        sanitized_chapter = selected_chapter.replace("'", "''")
        where_clause = f"chapter = '{sanitized_chapter}'"
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "where": where_clause  # Filter based on selected chapter
            }
        )
    except Exception as e:
        st.error(f"Error creating retriever with chapter filter: {e}")
        logging.error(f"Error creating retriever with chapter filter: {e}")
        return

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # User input for question
    question = st.text_input("Enter your question about the selected chapter:")
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                try:
                    # Use the RetrievalQA chain to get the answer and source documents
                    result = qa_chain({"query": question})
                    answer = result["result"]
                    source_documents = result["source_documents"]

                    st.subheader("Answer:")
                    st.write(answer)

                    if source_documents:
                        st.subheader("Source Documents:")
                        for doc in source_documents:
                            st.write(f"**Source:** {doc.metadata.get('source')}")
                            # Display a snippet of the document
                            snippet = doc.page_content[:200].replace('\n', ' ') + "..."
                            st.write(snippet)
                    else:
                        st.write("No source documents found.")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    logging.error(f"Error generating answer: {e}")
        else:
            st.warning("Please enter a question.")

if __name__ == '__main__':
    main()
