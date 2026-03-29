from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")  # ✅ correct
)

VECTORSTORE_PATH = "faiss_index"

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")   


# Load vectorstore
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

def retrieve(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    return docs


query = "what is this dataset about?"

def call_llm(query):
    docs = retrieve(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    Answer only on based of this context and if you don't know the answer say you don't know. Do not make up an answer.\n\n 
    Context: {context}
    Question: {query}
    Answer:"""
    
    response = llm.invoke(prompt)
    return response

print(call_llm(query))