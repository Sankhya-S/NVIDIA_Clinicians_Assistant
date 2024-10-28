from model_setup import setup_milvus, setup_embedding, setup_chat_model
from notes_processing import extract_text_from_pdf
from langchain.chains import RetrievalQA
from langchain.schema import Document  
from langchain.text_splitter import RecursiveCharacterTextSplitter

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")

# Set up Milvus, Embeddings, and Chat Models
embedding_model = setup_embedding()  # Initializes the NVIDIA embedding model
chat_model = setup_chat_model()  # Initializes the ChatNVIDIA model

# Input PDF file name and folder
pdf_path = "/raid/sivaks1/BasicRAG/NVIDIA_Clinicians_Assistant/MIMIC_notes_PDF/note_13180007-DS-14.pdf"

try:
    # Process the selected PDF
    document_text = extract_text_from_pdf(pdf_path)

    # Split the document text into smaller chunks (each <= 512 tokens)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    chunks = text_splitter.split_text(document_text)

    # Convert chunks into Document objects
    docs = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks]


    # Set up Milvus and add the document to the vector store
    milvus_vector_store = setup_milvus(embedding_model, docs, batch_size=10)

    # Set up Milvus retriever using the vector store
    retriever = milvus_vector_store.as_retriever()

    # Set up the RetrievalQA pipeline
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,  # Chat model for generating answers
        retriever=retriever,  # Milvus retriever for document retrieval
        return_source_documents=True  # Return source documents with responses
    )

    # input a question
    user_query = input("Ask a question based on the document: ")

    response = qa_chain.invoke({"query": user_query})

    # Extract result and source documents
    answer = response['result']  # The generated answer
    source_docs = response['source_documents']  # The retrieved source documents

    # Print the response
    print("Answer:")
    print(answer)
    # print("\nSource Documents:")
    # for doc in source_docs:
    #     print(doc.page_content)

except FileNotFoundError as e:
    print(e)