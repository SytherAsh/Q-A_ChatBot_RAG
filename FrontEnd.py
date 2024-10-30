import streamlit as st

# Page configuration
st.set_page_config(page_title="Chatbot Ollama", layout="wide", page_icon="🤖")

# Sidebar for additional options
with st.sidebar:
    st.button("+ New chat", key="new_chat")
    st.button("+ New prompt", key="new_prompt")
    st.text_input("Search...", key="search_bar")
    st.markdown("---")
    st.text("No data.")
    st.markdown("---")
    st.button("Import data", key="import_data")
    st.button("Export data", key="export_data")
    st.button("Settings", key="settings")

# Header
st.markdown("<h1 style='text-align: center; color: white;'>Chatbot Ollama</h1>", unsafe_allow_html=True)

# Main container for chat settings and responses
with st.container():
    st.write("### Model")
    model = st.selectbox("Select Model", ["Default (llama2:latest)", "GPT-4", "BERT"])
    
    st.write("### System Prompt")
    system_prompt = st.text_area(
        label="",
        value="You are Chatbot Ollama, a chatbot baked by a large language model. "
              "Follow the user's instructions carefully. Respond using markdown.",
        height=100
    )
    
    st.write("### Temperature")
    temperature = st.slider(
        label="",
        min_value=0.2,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."
    )
    
    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        st.write("Precise")
    with col3:
        st.write("Creative")

# Chat area
st.markdown("---")
st.write("Type a message or type `/` to select a prompt...")

# Input and output display
if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("Your message:", key="user_input")

if st.button("Send"):
    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        # Here you would call your chatbot model, e.g., with `model_response = get_chatbot_response(user_input)`
        model_response = "This is a response from Chatbot Ollama."
        st.session_state.conversation.append({"role": "bot", "content": model_response})

# Display chat history
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Chatbot Ollama:** {message['content']}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Chatbot Ollama is an advanced chatbot kit for Ollama models aiming to mimic ChatGPT’s interface and functionality.</p>", unsafe_allow_html=True)
