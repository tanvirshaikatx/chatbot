from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Streamlit framework
st.title('Langchain Demo With LLAMA2 (Ollama)')
input_text = st.text_input("Search the topic you want")

# Ollama LLama2 LLM
llm = Ollama(model="llama2")  # Ensure the model name matches the one in Ollama
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
