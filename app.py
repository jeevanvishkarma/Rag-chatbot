import streamlit as st
from main import call_llm as ask

# ✅ PAGE CONFIG
st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")
st.title("📄 RAG Chatbot ")
st.write("Ask questions about your documents")

# ✅ SIDEBAR
st.sidebar.title("Settings")
st.sidebar.info("This RAG chatbot uses your document embeddings with Grok API")

# ✅ MAIN INTERFACE
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input("🔍 Ask a question:")

with col2:
    submit = st.button("Submit", use_container_width=True)

# ✅ QUERY LOGIC
if submit and query:
    with st.spinner("🤔 Thinking..."):
        try:
            answer = ask(query)
            st.success("✅ Answer found!")
            st.write(answer)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ✅ FOOTER
st.divider()
st.caption("Powered by RAG + Grok API")
