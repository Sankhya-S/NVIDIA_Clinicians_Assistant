![NVIDID_Clinicians_Assistant](Image_Folder/Header_3.png)


# NVIDIA Clinicians Assistant

## Background
Healthcare professionals often face the challenge of preparing for patient meetings with minimal time, sometimes as little as 10 minutes. This limited preparation time can make it difficult to review and understand a patient's medical history thoroughly. However, the advancement of large language models (LLMs) and generative language models offers a promising solution to this issue.

## Project Concept
This project is an AI-powered virtual "clinical assistant" that leverages a Retrieval-Augmented Generation (RAG) pipeline using LangChain, NVIDIA embeddings, and Milvus as the vector store. The clinical assistant can efficiently sift through extensive patient data, intelligently search for relevant information, and answer questions about diagnoses, health histories, and more. It allows users to upload a PDF document, process and store document embeddings in Milvus, and ask questions based on the content of the document. This application aims to streamline the preparation process for healthcare professionals by providing quick and accurate summaries of health records and answering specific queries about individual patients.

## Objectives
1. Answering Health-Related Questions: The virtual assistant should be able to respond to questions related to a patient's health and diagnosis.
   
2. Summarizing Health History: It should summarize the patient's health history, emphasizing recent reports while understanding medical terminology and temporal effects.

### MIMIC-IV-Note Dataset
[Visit the MIMIC-IV Clinical Database](https://physionet.org/content/mimic-iv-note/2.2/)

The MIMIC-IV-Note dataset is a collection of deidentified free-text clinical notes linked to the MIMIC-IV clinical database. It is designed to advance research in clinical natural language processing (NLP) by addressing the scarcity of large-scale, open-access clinical text datasets.

#### This dataset includes:
- **331,794 deidentified discharge summaries** from **145,915 patients** admitted to the hospital and emergency department.
- **2,321,355 deidentified radiology reports** for **237,427 patients**.

The data is sourced from the Beth Israel Deaconess Medical Center in Boston, MA, USA, and all notes have undergone deidentification under the Health Insurance Portability and Accountability Act (HIPAA) Safe Harbor provision.

## Technologies Used

- **NVIDIA Embeddings**: Utilizes `nvidia/nv-embedqa-e5-v5` for embedding text.
- **Milvus Vector Store**: Uses Milvus to store document embeddings, enabling efficient retrieval for question answering.
- **Chat Model**: Uses `meta/Llama 3.1 70b nemotron` to generate answers based on retrieved document content.


## Evaluation Using Ragas

### Overview
Ragas is an open-source library designed for evaluating Large Language Model (LLM) applications, specifically **Retrieval Augmented Generation (RAG)** pipelines. This project uses Ragas to assess the performance of our clinical assistant chatbot and iteratively improve its architecture.

### Evaluation Workflow
Ragas allows us to:
1. Generate a test set of queries.
2. Evaluate pipeline performance using the `evaluate_pipeline` function and a set of metrics.
3. Compare answers to test set queries against the ground truth.

#### Metrics Evaluated
The following metrics are used to assess the chatbot's performance:
- **Context Precision**: Measures whether the retrieved documents are relevant to the query.
- **Recall**: Measures whether all relevant documents are retrieved.
- **Response Relevance**: Measures the alignment between the generated response and the ground truth.

### Architectures for Evaluation
We have developed and named four architectures to measure performance and improve the chatbot's effectiveness:
1. **Basic RAG**: The simplest implementation of a RAG pipeline.
2. **Advanced Document Splitting RAG**: Incorporates optimized document chunking strategies.
3. **Hybrid Search RAG**: Combines vector search with traditional keyword-based search.
4. **Advanced Document Splitting and Hybrid Search RAG**: Integrates advanced document splitting with hybrid search for maximum performance.



### Architectures
![NVIDID_Clinicians_Assistant](Image_Folder/RAG-Basic.png)
![NVIDID_Clinicians_Assistant](Image_Folder/RAG-ADParch.png)
![NVIDID_Clinicians_Assistant](Image_Folder/RAG-Hybrid.png)


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
