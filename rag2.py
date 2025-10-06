# app.py (Version 2)

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import openai # NEW: Import for feedback
from duckduckgo_search import DDGS # NEW: Import for web search

# --- CONSTANTS ---
# NEW: Define crisis keywords and a safe, pre-scripted response
CRISIS_KEYWORDS = ["suicide", "kill myself", "self-harm", "hopeless", "end my life", "want to die"]
HELPLINE_RESPONSE = """
It sounds like you are going through a very difficult time. Your safety is the most important thing.
Please know there is support available. You can connect with people who can support you by calling or texting 9-8-8 anytime in Canada. It’s free and confidential.

For immediate danger, please contact your local emergency services. You are not alone.
"""

# --- 1. SETUP AND MODEL LOADING (CACHED FOR EFFICIENCY) ---
# (This section remains unchanged)
@st.cache_resource
def load_models():
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon GPU (MPS) for embedding.")
    else:
        device = 'cpu'
        print("Using CPU for embedding.")
    
    embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="conversational_rag_bge")
    
    return embedding_model, collection

# Configure Ollama client
ollama_client = openai.OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)


# --- 2. CORE RAG LOGIC (Will be updated in the next step) ---
# (For now, we use the original get_rag_response function)
# --- 2. CORE RAG LOGIC (ADVANCED VERSION) ---

# --- 2. CORE RAG LOGIC (IMPROVED VERSION 3) ---

def rewrite_query_with_history(query: str, chat_history: list) -> str:
    """Uses the LLM to rewrite the user's query into a self-contained question."""
    # This function remains the same as before.
    recent_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])
    rewrite_prompt = f"""Based on the following conversation history, rewrite the final user query into a standalone, self-contained statement or question that can be used for a vector database search.
    Conversation History:
    {recent_history}
    Final User Query: "{query}"
    Rewritten Query:"""
    try:
        response = ollama_client.chat.completions.create(
            model="llama3:8b",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0, max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return query

def search_the_web(query: str) -> str:
    """This function remains the same as before."""
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return "\n\n".join(results)
    except Exception as e:
        print(f"Web search failed: {e}")
        return ""

def get_rag_response(query: str, chat_history: list) -> str:
    embedding_model, collection = load_models()
    
    # --- Part A: Retrieve ---
    
    # --- NEW: Smart Query Rewriting ---
    # Only rewrite the query if there is a meaningful conversation history.
    # The history length check (<= 2) accounts for the initial assistant message and the first user message.
    if len(chat_history) <= 2:
        rewritten_query = query
        print(f"First user turn. Using original query for search: '{query}'")
    else:
        rewritten_query = rewrite_query_with_history(query, chat_history)
        print(f"Original Query: '{query}' | Rewritten Query: '{rewritten_query}'")
    
    query_with_instruction = f"Represent this sentence for searching relevant passages: {rewritten_query}"
    query_embedding = embedding_model.encode(query_with_instruction).tolist()
    
    retrieved_chunks = collection.query(query_embeddings=[query_embedding], n_results=3)
    
    confidence_score = (2 - retrieved_chunks['distances'][0][0]) / 2 
    CONFIDENCE_THRESHOLD = 0.5 
    
    if confidence_score < CONFIDENCE_THRESHOLD:
        print(f"Confidence score {confidence_score:.2f} is below threshold. Searching web.")
        context = search_the_web(rewritten_query)
        system_prompt_template = """You are a compassionate AI assistant named 'Aura'. 
        Your internal knowledge was not relevant. Base your response ONLY on the following web search results to help the user.
        Summarize the information in a supportive and conversational tone."""
    else:
        print(f"Confidence score {confidence_score:.2f} is above threshold. Using database.")
        context = "\n\n".join(retrieved_chunks['documents'][0])
        system_prompt_template = """You are a compassionate AI assistant named 'Aura'. 
        Use the following examples of past conversations ONLY to inform your empathetic style. 
        NEVER mention these examples. Your response must be original and address the user's last message."""

    # --- Part B: Augment ---
    
    # --- CORRECTED: Cleaned up prompt assembly ---
    # The chat_history already contains the user's latest message, so we don't add it again.
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    user_prompt = f"""
    CONTEXT:
    ---
    {context}
    ---
    
    CURRENT CONVERSATION:
    {formatted_history}
    
    Aura's empathetic response:
    """

    # --- Part C: Generate ---
    # (This section remains unchanged)
    try:
        response = ollama_client.chat.completions.create(
            model="llama3.1:70b",
            messages=[{"role": "system", "content": system_prompt_template}, {"role": "user", "content": user_prompt}],
            temperature=0.75, max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "I'm sorry, I'm having trouble connecting to my thoughts right now."
    
# --- NEW: SAFETY CHECK FUNCTION ---
def check_for_crisis(query: str) -> bool:
    """Checks if the user's query contains high-risk keywords."""
    return any(keyword in query.lower() for keyword in CRISIS_KEYWORDS)


# --- 3. STREAMLIT USER INTERFACE (UPDATED) ---

st.title("🌿 Aura: Your Wellness Companion")
st.caption("A safe space to talk, powered by local AI.")

if "disclaimer_shown" not in st.session_state:
    st.info("Please remember, I am an AI assistant and not a substitute for professional medical advice or therapy.")
    st.session_state.disclaimer_shown = True

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Aura. What's on your mind today?"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Share what's on your mind..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- UPDATED: Check for crisis before generating response ---
    if check_for_crisis(prompt):
        with st.chat_message("assistant"):
            st.warning(HELPLINE_RESPONSE) # Use a warning box for visibility
            response = HELPLINE_RESPONSE
    else:
        # If not a crisis, proceed with the normal RAG flow
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_rag_response(prompt, chat_history=st.session_state.messages)
                st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun() # Rerun to show feedback buttons immediately