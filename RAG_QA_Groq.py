import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
load_dotenv()

# Load environment variables
groq_api = os.getenv("GROQ_API")
os.environ['GOOGLE_API_KEY'] = os.getenv("Gemma")
huggingFace = os.getenv("HuggingFace")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingFace

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure text extraction is successful
                text += page_text + "\n"
    return text.strip()  # Trim whitespace

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=huggingface_embeddings)
    vector_store.save_local("faiss_index")


def user_input(user_question):
    global retrieval_qa_instance
    
    # Load embeddings and vector store
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_db = FAISS.load_local("faiss_index", huggingface_embeddings,allow_dangerous_deserialization=True)

    retriever=vector_db.as_retriever(search_type="similarity",search_kwargs={"k":3})


    llm=ChatGroq(groq_api_key=groq_api,model_name='Gemma-7b-it')

    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    answer=chain.invoke({"input": user_question})
    # print(answer['answer'])
    st.write(answer['answer'])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Mistral💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()