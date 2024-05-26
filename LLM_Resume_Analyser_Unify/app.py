import PyPDF2
from PyPDF2 import PdfReader
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from data.unify_endpoints_data import model_provider, dynamic_provider
from io import StringIO
import pandas as pd
import base64

def split_text(text): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        length_function=len)

    chunks = text_splitter.split_text(text=text)
    return chunks


def faiss_vector_storage(chunks):
    vector_store = None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def LLM_QA(vector_store):
    # compares the query and chunks, enabling the selection of the top 'K' most similar chunks based on their similarity scores.
    docs = vector_store.similarity_search(query=analyze, k=3)

    # creates an OpenAI object, using the ChatGPT 3.5 Turbo model
    llm = ChatOpenAI(model='gpt-3.5-turbo', api_key=openai_api_key)

    # question-answering (QA) pipeline, making use of the load_qa_chain function
    chain = load_qa_chain(llm=llm, chain_type='stuff')

    response = chain.run(input_documents=docs, question=analyze)
    return response

def process_inputs():
    if not st.session_state.unify_api_key or not st.session_state.endpoint or not st.session_state.pdf_docs:
        st.warning("Please enter the missing fields and upload your pdf document(s)")


def landing_page():
    st.set_page_config("LLM Resume Analyser", page_icon="🚀")
    st.title("LLM Resume Analyser 🚀")
    st.text("Improve your resume with the power of LLMs")
    st.write('''
    Usage: 
    1. Input your **Unify API Key.** If you don’t have one yet, log in to the [console](https://console.unify.ai/) to get yours.
    2. Choose your Endpoint by selecting your **Model and Provider ID**.
    3. Upload your **Resume** and select or upload your **Job Description**.
    4. Get insights about how you can improve your Resume and be a better match for the role. 
    ''')

    with st.sidebar:
        # input for Unify API Key
        unify_api_key = st.text_input("Unify API Key*", type="password",placeholder="Enter Unify API Key", args=("Unify Key ",))
        # Model and provider selection
        model_name = st.selectbox("Select Model*", options=model_provider.keys(), index=20, placeholder="Model", args=("Model",))
        if st.toggle("Enable Dynamic Routing"):
            provider_name = st.selectbox("Select a Provider*", options=dynamic_provider, placeholder="Provider", args=("Provider",))
        else:
            provider_name = st.selectbox("Select a Provider*", options=model_provider[model_name], placeholder="Provider", args=("Provider",))
        endpoint = f"{model_name}@{provider_name}"
        # Document uploader
        uploaded_file = st.file_uploader(label="Upload your Resume (PDF)*", type="pdf", accept_multiple_files=False)
        #st.button("Submit Document(s)", on_click=process_inputs)
        job_description = st.selectbox(label = "Select Job Description*", 
                                       
                                       options=model_provider[model_name] , #change this to job description suggestions

                                        placeholder="Job Description", args=("Job Description",))
        uploaded_job_description = st.file_uploader(label="Upload your Job Description (PDF)*", type="pdf", accept_multiple_files=False)
        return uploaded_file

def resume_insights(uploaded_file):
    if uploaded_file is not None:

        #Indexing flow
        """" 
        1. Collect your knowledge base documents.
        2. Extract plain text from the documents.
        3. Split the text into chunks.
        4. Use an embedding model to transform each chunk into an embedding vector.
        5. Save the embedding vectors into a vector store along with the index (a mapping 
        from an embedding vector back to a document reference).
        """
        pdfReader = PyPDF2.PdfReader(uploaded_file)
        pageobj = pdfReader.pages[len(pdfReader.pages)-1]
        resulttext = pageobj.extract_text()
        chunks = split_text(resulttext)
        st.write(chunks)
        vector_store = faiss_vector_storage(chunks)
        st.write(vector_store)

        # Query flow
        """ 
        1. Clean and summarize the user query.
        2. Use the same embedding model to vectorize the query.
        3. Query the vector store for top K documents that are the most similar to the question vector.
        4. Pass the retrieved chunks of text and the question to the LLM, instructing it to inspect the text and answer the question.
        This approach, known as retrieval-augmented generation (RAG), uses embedding vectors and similarity search to pre-filter relevant text before passing it to the LLM,
        making the process more efficient and cost-effective.
        """

    #chain = load_summarize_chain(llm=model_name, chain_type="stuff")
    #chain.run(docs)

def main():
    uploaded_file = landing_page()
    resume_insights(uploaded_file)

if __name__ == "__main__":
    main()


# if file_upload:
#     try:
#         text = extract_text_from_file(file_upload)
        
#     # TODO
#     # ?  chain = load_qa_chain(llm=llm, chain_type='stuff')
#     # ?  with st.expander("analysis"):
#     # ?      st.markdown("**analysis **")
#     # ?       st.markdown(analyisis anwser)
        
#     except ValueError as e:
#         st.error(str(e))

