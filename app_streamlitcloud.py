import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import os

# Get the Hugging Face API token from the environment variable
huggingface_api_token = os.getenv('HUGGINGFACE_API_TOKEN')

# Load the FAISS index and documents
@st.cache_resource
def load_resources():
    index = faiss.read_index('./faiss_index.index')
    with open('./documents.pkl', 'rb') as f:
        documents = pickle.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, documents, model

index, documents, model = load_resources()

# Initialize the HuggingFace LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 0.2, "max_new_tokens": 512, "return_full_text": False},
    huggingfacehub_api_token=huggingface_api_token
)

# Define the prompt template
prompt_template = PromptTemplate(
    template="You are an AI assistant helping to answer FAQs. Use the following context to provide an answer. If the user asks an irrelevant question or greets, display appropriate message.\n\nContext:\n{context}\n\nQuery:\n{query}\nAnswer:",
    input_variables=["context", "query"]
)

# Function to get suggestions
def get_suggestions(user_input, top_k=3):
    if user_input.strip() == "":
        return []
    input_embedding = model.encode([user_input])
    distances, indices = index.search(np.array(input_embedding), top_k)
    suggested_questions = [documents[i]['question'] for i in indices[0]]
    return suggested_questions

# Function to retrieve relevant context
def get_relevant_context(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [documents[i] for i in indices[0]]
    context = " ".join([doc['answer'] for doc in results])
    return context

# Function to generate answer with LLM
def generate_answer_with_llm(query):
    context = get_relevant_context(query)
    formatted_prompt = prompt_template.format(context=context, query=query)
    response = llm(formatted_prompt)
    return response

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": """Welcome to the FAQ Wizard! 😊
Feel free to type your query below to receive some helpful question suggestions. You can click on any suggestion that catches your eye, or if you prefer, type your own question. Happy chatting!"""}
    ]
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = False
if "current_suggestions" not in st.session_state:
    st.session_state.current_suggestions = []

# Main chat area
st.title("FAQ Wizard Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your question here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display response
    with st.chat_message("assistant"):
        response = generate_answer_with_llm(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Generate new suggestions
    st.session_state.current_suggestions = get_suggestions(prompt)
    st.session_state.show_suggestions = True
    st.rerun()

# Sidebar for suggestions
with st.sidebar:
    st.header("Suggestions")
    if st.session_state.show_suggestions and st.session_state.current_suggestions:
        st.subheader("You might also want to ask:")
        for suggestion in st.session_state.current_suggestions:
            if st.button(suggestion, key=suggestion):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                with st.chat_message("user"):
                    st.markdown(suggestion)
                with st.chat_message("assistant"):
                    response = generate_answer_with_llm(suggestion)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                # Generate new suggestions based on the selected suggestion
                st.session_state.current_suggestions = get_suggestions(suggestion)
                st.rerun()
