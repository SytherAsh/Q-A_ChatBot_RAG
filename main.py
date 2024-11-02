import streamlit as st
import time
import numpy as np
import pandas as pd
from Groq import get_response, display_message, format_response 
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
    
    # st.button("+ New chat", key="new_chat")
    # st.button("+ New prompt", key="new_prompt")
    st.markdown("---")
    # st.text("No data.")
    # if st.button("Import data", key="import_data"):
    #     st.write("Data imported")

    #     uploaded_files = st.file_uploader(
    #         "Choose a Document", accept_multiple_files=True
    #     )
    #     for uploaded_file in uploaded_files:
    #         bytes_data = uploaded_file.read()
    #         st.write("filename:", uploaded_file.name)
    #         st.write(bytes_data)
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


    st.button("Settings", key="settings")

#! Header
st.markdown("<h1 style='text-align: center;'>GroqBot 🤖</h1>", unsafe_allow_html=True)


    
#! Chat area
with st.chat_message(avatar="🤖", name="Chatbot Ollama"):
    user_input = st.chat_input("Type a message or type ")

#! Input and output display
def stream_data(Groq_formatted_response):
    for word in Groq_formatted_response.split(" "):
        yield word + " "
        time.sleep(0.02)
        
if user_input:
    Groq_response = get_response(user_input, model_choice,prompt_choice)
    Groq_formatted_response = format_response(Groq_response)
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
   
# with st.container():
#     st.write("### System Prompt")
#     system_prompt = st.text_area(
#         label="",
#         value="You are Chatbot Ollama, a chatbot baked by a large language model. "
#               "Follow the user's instructions carefully. Respond using markdown.",
#         height=100
#     )
#  if st.button("Prompt",use_container_width=True):
#         st.write("### System Prompt")
#         system_prompt = st.text_area(
#             label="",
#             value="You are Chatbot Ollama, a chatbot baked by a large language model. "
#                 "Follow the user's instructions carefully. Respond using markdown.",
#             height=100
#         )

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
  
    # st.write("### Temperature")
    # temperature = st.slider(
    #     label="",
    #     min_value=0.2,
    #     max_value=1.0,
    #     value=1.0,
    #     step=0.1,
    #     help="Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."
    # )
    