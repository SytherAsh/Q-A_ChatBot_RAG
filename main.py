import streamlit as st
import time
from Groq import get_response,format_response 
#! Page configuration
st.set_page_config(page_title="GroqBot", layout="wide", page_icon="🤖")

#!conversation for download
if "conversation" not in st.session_state:
    st.session_state.conversation = []


#! Custom CSS for styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    .stTextInput, .stTextArea, .stButton, .stSelectbox, .stSlider {
        border-radius: 10px;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #f1f3f4;
        color: #333;
    }
    .stButton button {
        background: linear-gradient(to right, #4CAF50, #81C784);
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition: background 0.3s;
    }
    .stButton button:hover {
        background: linear-gradient(to right, #388E3C, #66BB6A);
    }
    .stMarkdown h1 {
        font-family: 'Arial', sans-serif;
        color: #007BFF;
    }
    .stMarkdown p {
        text-align: center;
        color: #6c757d;
    }
    .stMarkdown hr {
        border: 1px solid #dee2e6;
    }
    .stContainer {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

#! Sidebar for additional options
with st.sidebar:
    model_choice = st.selectbox("Model", ["Gemma-7b-it","mixtral-8x7b-32768"])
    prompt_choice = st.selectbox("Prompt", ["Chatgpt","Google","OpenAI"])
    prompt_new=st.sidebar.toggle("CustomPrompt", key="prompt")

    st.markdown("---")
    temperature = st.slider(
        label="Temperature",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="""Higher values like 0.8 will make the output more random, 
        while lower values like 0.2 will make it more focused and deterministic."""
    )

   
    import_button = st.sidebar.button("Import", key="Import")
 
    if st.sidebar.button("Export", key="Export"):
        conversation_text = ""
        for message in st.session_state.conversation:
            if message["role"] == "user":
                conversation_text += f"**SAWASH:** {message['content']}\n"
            else:
                conversation_text += f"**GroqBot:** {message['content']}\n"
        conversation_text += "\n"
        st.download_button("Download", conversation_text, "conversation.txt", "text/plain",use_container_width=True)


#! Header
st.markdown("<h1 style='text-align: center;'>GroqBot 🤖</h1>", unsafe_allow_html=True)

    
#! Chat area
with st.chat_message(avatar="🤖", name="Chatbot Ollama"):
    user_input = st.chat_input("Type a message or type ")
#!Custom Prompt
custom_prompt=""
if prompt_new:
    st.write("### System Prompt")
    custom_prompt = st.text_area(
        label="",
        value="You are Chatbot Groq, a chatbot baked by a large language model. "
            "Follow the user's instructions carefully. Respond using markdown.",
        height=100
    )
#! Input and output display
def stream_data(Groq_formatted_response):
    for word in Groq_formatted_response.split(" "):
        yield word + " "
        time.sleep(0.02)

if user_input and not prompt_new:
    Groq_response = get_response(user_input, model_choice,prompt_choice,temperature)
    Groq_formatted_response = format_response(Groq_response)
    # st.write(prompt_choice)
    st.write_stream(stream_data(Groq_formatted_response))    
    st.session_state.conversation.append({"role": "user", "content": user_input})
    model_response = Groq_response.content
    st.session_state.conversation.append({"role": "bot", "content": model_response})
elif user_input and  prompt_new:
    Groq_response = get_response(user_input, model_choice,custom_prompt,temperature)
    Groq_formatted_response = format_response(Groq_response)
    # st.write(custom_prompt)
    st.write_stream(stream_data(Groq_formatted_response))    
    st.session_state.conversation.append({"role": "user", "content": user_input})
    model_response = Groq_response.content
    st.session_state.conversation.append({"role": "bot", "content": model_response})
    
   

st.markdown("---")


col2, col3,col4 = st.columns([1,1,1])
with col2:
   if st.button("Clear Chat",use_container_width=True):
         st.session_state.conversation = []
         st.session_state.conversation.clear()
         st.write("Chat cleared")
with col4:
    show_botton = st.button("Show Chat",use_container_width=True)
   


#! Upload Document
if import_button:
    uploaded_files = st.file_uploader(
            "Choose a Document", accept_multiple_files=True
        )
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)

# #! Display chat history
if show_botton:
    st.markdown("<h2 style='text-align: center;'>Conversation</h2>", unsafe_allow_html=True)
    conversation_area = st.container()
    conversation_area.empty()
    for message in st.session_state.conversation:
        if message["role"] == "user":
            st.write(f"**SAWASH:** {message['content']}")
        else:
            st.write(f"**GroqBot:** {message['content']}")



#! Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>GroqBot is an advanced chatbot kit for Multiple models aiming to mimic ChatGPT’s interface and functionality.</p>", unsafe_allow_html=True)


# Main container for chat settings and responses
  
