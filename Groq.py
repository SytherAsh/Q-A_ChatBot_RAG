from langchain_groq import ChatGroq
import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

groq_api=os.getenv("GROQ_API")
os.environ['GOOGLE_API_KEY']=os.getenv("Gemma")



def get_response(user_input, model_name,system_prompt,temp):
    llm1=ChatGroq(groq_api_key=groq_api,model_name='Gemma-7b-it',temperature=temp)
    llm2 = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=temp,
        max_retries=2,
        api_key=groq_api
    )    
    if system_prompt =="Chatgpt":
        system_prompt="""
                    System Prompt: You are an AI assistant designed to assist and communicate with humans. 
                    Your primary goal is to provide accurate and helpful responses to user queries. 
                    You have been trained on a massive dataset of text from the internet and can generate human-like responses. 
                    Please respond to the user's input in a clear and concise manner, using proper grammar and spelling.

        """
    elif system_prompt=="Google":
        system_prompt=""""You are a conversational AI designed to engage in natural-sounding conversations with humans. 
                    our goal is to respond to user input in a way that simulates human-like conversation,
                    using context and understanding to generate relevant and coherent responses.
                    Please use a friendly and approachable tone, and avoid using overly formal or robotic language.
        """
    elif system_prompt=="OpenAI":
        system_prompt=""""
                    You are a highly advanced language model designed to generate human-like text based on user input. 
                    Your goal is to produce responses that are not only accurate but also creative, engaging, and relevant to the user's query. 
                    Please use your vast knowledge and understanding of language to generate responses that are both informative and entertaining.
                    Feel free to use humor, anecdotes, and creative storytelling to make your responses more engaging and memorable.
        """
    else:
        system_prompt=system_prompt
    
    messages = [
        ("system", system_prompt),
        ("human", user_input),
    ]

    if model_name == "Gemma-7b-it":
        result = llm1.invoke(messages)
    else:
        result = llm2.invoke(messages)
    # print(system_prompt)
    # print("_____")
    return result

        
def format_response(ai_message):
    # This function formats and prepares the response for display
    content = ai_message.content
    response_metadata = ai_message.response_metadata

    formatted_output = {
        "content": content,
        "metadata": {
            "token_usage": response_metadata.get('token_usage', {}),
            "model_info": {k: v for k, v in response_metadata.items() if k != 'token_usage'}
        }
    }
    formatted_string = f"Response:\n{formatted_output['content']}\n\nMetadata:\n"
    
    for key, value in formatted_output['metadata'].items():
        formatted_string += f"{key} \n"
        for v in value:
            formatted_string += f"\t{v} : {value[v]}\n"

    return formatted_string
    

# groq_res=get_response("What is gpt-4?","Gemma-7b-it","I want to know about gpt-4")
# print(groq_res)