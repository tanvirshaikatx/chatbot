# # groq api=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
# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# import time

# from dotenv import load_dotenv
# load_dotenv()

# ## load the Groq API key
# groq_api_key=os.environ['GROQ_API_KEY']

# if "vector" not in st.session_state:
#     st.session_state.embeddings=OllamaEmbeddings()
#     st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
#     st.session_state.docs=st.session_state.loader.load()

#     st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#     st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
#     st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

# st.title("Chat with Gr")
# llm=ChatGroq(groq_api_key=groq_api_key,
#              model_name="mixtral-8x7b-32768")

# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )
# document_chain = create_stuff_documents_chain(llm, prompt)
# retriever = st.session_state.vectors.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# prompt=st.text_input("Input you prompt here")

# if prompt:
#     start=time.process_time()
#     response=retrieval_chain.invoke({"input":prompt})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")




#------------------- modified==================

# groq api=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
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

# Title of the app
st.title("Chat with Groq")

# Input for URL
url = st.text_input("Enter the URL you want to chat with:", placeholder="https://example.com")

# Proceed only if URL is provided
if url:
    if "vector" not in st.session_state or st.session_state.get("url") != url:
        st.session_state.url = url
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = WebBaseLoader(url)
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

    prompt = ChatPromptTemplate.from_template(
    """
    You are an intelligent chatbot designed to answer questions accurately based on the provided context.
    Follow these guidelines to deliver the best user experience:
    
    1. Answer questions precisely using the context given.
    2. If asked for a summary, provide a concise yet comprehensive summary of the context.
    3. If the question is about complex topics, explain them in simpler terms for better understanding.
    4. If a comparison is requested, compare relevant information present in the context.
    5. If the user asks for a list (e.g., steps, pros and cons, key points), provide a well-organized list.
    6. If the question requires an explanation or example, provide detailed and clear explanations or examples.
    7. If multiple interpretations exist, mention them clearly.
    8. If the user asks for your opinion, inform them that you are an AI language model and provide objective insights from the context.
    9. Always cite specific sections or quotes from the context to support your answer.
    10. If the information isn't available in the context, clearly state that the answer isn't present in the provided material.
    11. if they ask who made you. tell him your creator is Tanvir Shaikat
    <context>
    {context}
    <context>
    
    Question: {input}
    
    Provide the most accurate and relevant response to the above question, following the guidelines mentioned.
    """
)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    prompt = st.text_input("Input your prompt here")

    if prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
else:
    st.warning("Please enter a URL to begin chatting.")
