from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def initialize_llm():
    """Initialize the LLM with a prompt template"""
    # For testing with a smaller model (uncomment if you don't want to use Mistral)
    # llm = CTransformers(
    #     model="TheBloke/Llama-2-7B-Chat-GGML",
    #     model_type="llama",
    #     config={'max_new_tokens': 256, 'temperature': 0.01}
    # )
    
    # Using Mistral-7B (requires adequate hardware)
    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GGML",
        model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        model_type="mistral",
        config={'max_new_tokens': 512, 'temperature': 0.1, 'context_length': 2048}
    )
    
    # Define prompt template for RAG
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    return llm_chain

def generate_response(llm_chain, question, relevant_docs):
    """Generate a response using the LLM with retrieved context"""
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    response = llm_chain.run({"context": context, "question": question})