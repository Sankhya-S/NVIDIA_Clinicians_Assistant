# NVIDIA_Clinicians_Assistant

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using LangChain, NVIDIA embeddings, and Milvus as the vector store. It allows users to upload a PDF document, process and store document embeddings in Milvus, and ask questions based on the content of the document.


## Technologies Used

- **NVIDIA Embeddings**: Utilizes `nvidia/nv-embedqa-e5-v5` for embedding text.
- **Milvus Vector Store**: Uses Milvus to store document embeddings, enabling efficient retrieval for question answering.
- **Chat Model**: Uses `meta/llama-3.1-8b-instruct` to generate answers based on retrieved document content.


## Setup Steps

Follow these steps to get the chatbot up and running in less than 5 minutes:

### 1. Clone this repository 


### 2. Start NVIDIA Containers: 
Use the script_start.sh script to start the embedding and chat model containers

```bash
./script_start.sh true true true
```
First true initializes the Milvus database.
Second true starts the NVIDIA Embedding Model service.
Third true starts the NVIDIA Chat Model service.

### 3. Upload PDF Documents: 
Place the PDF documents you wish to process in the `./MIMIC_notes_PDF/ ` folder

### 4. Run the pipeline

```bash
python basic_RAG.py
```