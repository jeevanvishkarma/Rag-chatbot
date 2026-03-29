from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import os
import json


PROCESSED_FILES_PATH = "processed_files.json"
VECTORSTORE_PATH = "faiss_index"


def load_processed_files():
    """Load list of already processed files"""
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, 'r') as f:
            return json.load(f)
    return []


def save_processed_files(files):
    """Save list of processed files"""
    with open(PROCESSED_FILES_PATH, 'w') as f:
        json.dump(files, f)


def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


def get_new_pdfs(folder_path):
    """Get only new PDF files that haven't been processed"""
    processed = load_processed_files()
    new_pdfs = []
    
    for file in os.listdir(folder_path):
        if file.endswith('.pdf') and file not in processed:
            file_path = os.path.join(folder_path, file)
            new_pdfs.append((file, file_path))
    
    return new_pdfs


def chunk_document(document, chunk_size=500, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_documents(document)
    return chunks


def create_or_update_embeddings(chunks, persist_directory=VECTORSTORE_PATH):
    """Create new or update existing vectorstore"""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # If vectorstore exists, load and add new chunks
    if os.path.exists(persist_directory):
        print("Adding to existing vectorstore...")
        vectorstore = FAISS.load_local(persist_directory, embeddings)
        vectorstore.add_documents(chunks)
    else:
        print("Creating new vectorstore...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
    
    vectorstore.save_local(persist_directory)
    return vectorstore


if __name__ == "__main__":
    folder_path = r"C:\Users\jeeva\OneDrive\Desktop\Rag_Chatbot\Rag-chatbot\data"
    
    # Get only new PDFs
    new_pdfs = get_new_pdfs(folder_path)
    
    if new_pdfs:
        print(f"Found {len(new_pdfs)} new PDF(s)")
        all_documents = []
        
        for file_name, file_path in new_pdfs:
            print(f"Loading: {file_name}")
            try:
                docs = load_pdf(file_path)
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        
        if all_documents:
            chunks = chunk_document(all_documents)
            vectorstore = create_or_update_embeddings(chunks)
            
            # Update processed files list
            processed = load_processed_files()
            processed.extend([f[0] for f in new_pdfs])
            save_processed_files(processed)
            
            print("Done! Vectorstore updated with new PDFs")
    else:
        print("No new PDFs to process")