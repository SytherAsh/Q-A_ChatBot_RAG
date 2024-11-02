from langchain_groq import ChatGroq
import os

from dotenv import load_dotenv
load_dotenv()

groq_api=os.getenv("GROQ_API")
os.environ['GOOGLE_API_KEY']=os.getenv("Gemma")
# groq_api="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"
# os.environ['GOOGLE_API_KEY']="AIzaSyDXfGOl-rOfEQYRo47SbR-dS0IW6v49CBE"

llm1=ChatGroq(groq_api_key=groq_api,model_name='Gemma-7b-it')
llm2 = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=1,
    max_retries=2,
    api_key=groq_api
)


def get_response(user_input, model_name,system_prompt):
    
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
        system_prompt="""
                    You are an advanced AI model capable of providing expert-level answers on a variety of topics, including but not limited to technology, science, history, current events, and general knowledge. 
                    When answering:
                    Be clear and detailed: Provide an answer that is easy to understand, with sufficient context or background information when necessary.
                    Be precise and factual: Ensure all information is accurate, up-to-date, and well-researched.
                    Use references when applicable: If a question requires deeper context or supporting information, cite reputable sources or refer to relevant studies, papers, or examples.
                    Adapt to the questions complexity: For technical or complex queries, provide a detailed explanation and, if helpful, include examples, analogies, or breakdowns of key concepts.
                    Stay professional and neutral: Offer balanced perspectives when answering subjective or opinion-based questions.
                    Respond fully to each question and let me know if further clarification or details are needed."""
    
    messages = [
        ("system", system_prompt),
        ("human", user_input),
    ]

    if model_name == "Gemma-7b-it":
        result = llm1.invoke(messages)
    else:
        result = llm2.invoke(messages)

    return result

def display_message(ai_message):
    # Extract content and metadata
    content = ai_message.content
    response_metadata = ai_message.response_metadata

    # Display the main content
    print(content)
    
    # Display metadata in a structured format
    print("\nMetadata:")
    
    # Token usage details, if available
    if 'token_usage' in response_metadata:
        print("Token Usage:")
        for key, value in response_metadata['token_usage'].items():
            print(f"  {key.replace('_', ' ').capitalize()}: {value}")

    # General metadata details
    print("\nModel Information:")
    for key, value in response_metadata.items():
        if key == 'token_usage':
            continue
        print(f"  {key.replace('_', ' ').capitalize()}: {value}")
        
        
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
    

