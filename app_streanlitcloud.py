import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from huggingface_hub import HuggingFaceHub
from langchain.prompts import PromptTemplate

# Load the FAISS index and documents
index = faiss.read_index('/kaggle/working/faiss_index.index')
with open('/kaggle/working/documents.pkl', 'rb') as f:
    documents = pickle.load(f)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get the Hugging Face API token from Streamlit secrets
huggingface_api_token = st.secrets["huggingface"]["api_token"]

# Initialize the HuggingFace LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 0.2, "max_new_tokens": 512, "return_full_text": False},
    huggingfacehub_api_token=huggingface_api_token  # Use the token from Streamlit secrets
)

prompt_template = PromptTemplate(
    template="You are an AI assistant helping to answer user queries. Use the following context to provide an answer.\n\nContext:\n{context}\n\nQuery:\n{query}\nAnswer:",
    input_variables=["context", "query"]
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input for user prompt
user_input = st.text_input("Type your question:")

# Display suggestions as the user types
if user_input:
    suggestions = get_suggestions(user_input)
    if suggestions:
        st.sidebar.subheader("Suggestions:")
        for suggestion in suggestions:
            st.sidebar.write(suggestion)

# Generate response on Enter
if st.button("Submit"):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response using the LLM
        response = generate_answer_with_llm(user_input)

        # Append assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
