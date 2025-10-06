# app.py

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import openai # We use the openai library to interface with Ollama's API

# --- 1. SETUP AND MODEL LOADING (CACHED FOR EFFICIENCY) ---

@st.cache_resource
def load_models():
    """
    Loads the embedding model and connects to the ChromaDB collection.
    This function is cached so it only runs once.
    """
    # Set device for Apple Silicon GPU
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon GPU (MPS) for embedding.")
    else:
        device = 'cpu'
        print("Using CPU for embedding.")
    
    # Load the powerful BGE embedding model
    embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    
    # Connect to the persistent ChromaDB database
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="conversational_rag_bge")
    
    return embedding_model, collection

# Configure the client to connect to the local Ollama server
ollama_client = openai.OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', # Required, but can be any string
)


# --- 2. CORE RAG LOGIC ---

def get_rag_response(query: str, chat_history: list) -> str:
    """
    Performs the full Retrieve-Augment-Generate (RAG) process using a local LLM.
    """
    embedding_model, collection = load_models()
    
    # --- Part A: Retrieve ---
    query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
    query_embedding = embedding_model.encode(query_with_instruction).tolist()
    
    # Find the 3 most relevant conversational chunks from the database
    retrieved_chunks = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    retrieved_examples = "\n\n".join(retrieved_chunks['documents'][0])

    # --- Part B: Augment ---
    system_prompt = """You are a compassionate, non-judgmental AI assistant named 'Aura', focused on mental wellness. 
    Your role is to listen, provide comfort, and offer gentle, supportive guidance based on the conversation.
    Use the following examples of past successful conversations ONLY to inform your tone and empathetic style. 
    NEVER mention that you are using examples. Your response must be original and directly address the user's last message."""

    # Format conversation history for the prompt
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    # Create the final prompt for the LLM
    user_prompt = f"""
    HERE ARE EXAMPLES OF EMPATHETIC RESPONSES FOR INSPIRATION:
    ---
    {retrieved_examples}
    ---
    
    CURRENT CONVERSATION:
    {formatted_history}
    user: {query}
    
    Aura's empathetic response:
    """

    # --- Part C: Generate ---
    try:
        response = ollama_client.chat.completions.create(
            model="llama3.1:70b", # Ensure you have pulled this model with Ollama
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.75,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred while connecting to Ollama: {e}")
        return "I'm sorry, I'm having trouble connecting to my thoughts right now. Please make sure Ollama is running."


# --- 3. STREAMLIT USER INTERFACE ---

st.title("🌿 Aura: Your Wellness Companion")
st.caption("A safe space to talk, powered by local AI.")

# Initial disclaimer message
if "disclaimer_shown" not in st.session_state:
    st.info("Please remember, I am an AI assistant and not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")
    st.session_state.disclaimer_shown = True

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Aura. I'm here to listen. What's on your mind today?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Share what's on your mind..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_rag_response(prompt, chat_history=st.session_state.messages)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})