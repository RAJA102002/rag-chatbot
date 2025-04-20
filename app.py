import streamlit as st
from utils.document_processor import process_uploaded_files
from utils.embeddings import initialize_embeddings, create_vector_store, get_retriever
from utils.llm_utils import initialize_llm, generate_response
import os
import time

def initialize_session_state():
    """Initialize session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'llm_chain' not in st.session_state:
        st.session_state.llm_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False

def display_chat_history():
    """Display chat history in the Streamlit app"""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.title("RAG-based Chatbot with Open-Source Models")
    
    initialize_session_state()
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("Configuration")
        
        uploaded_files = st.file_uploader(
            "Upload documents (PDF or text)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        process_button = st.button("Process Documents")
        
        if process_button and uploaded_files:
            with st.spinner("Processing documents..."):
                # Process uploaded files
                chunks = process_uploaded_files(uploaded_files)
                
                # Initialize embeddings and vector store
                embeddings = initialize_embeddings()
                st.session_state.vector_store = create_vector_store(chunks, embeddings)
                st.session_state.documents_processed = True
                
                # Initialize LLM
                st.session_state.llm_chain = initialize_llm()
                
            st.success("Documents processed successfully!")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This chatbot uses RAG (Retrieval-Augmented Generation) with open-source models.")
    
    # Main chat interface
    if st.session_state.documents_processed:
        display_chat_history()
        
        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve relevant chunks
                    retriever = get_retriever(st.session_state.vector_store)
                    relevant_docs = retriever.get_relevant_documents(prompt)
                    
                    # Generate response using LLM
                    response = generate_response(
                        prompt,
                        relevant_docs
                    )
                    
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        st.info("Please upload and process documents to start chatting.")

if __name__ == "__main__":
    main()