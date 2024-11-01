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
    
    if not system_prompt:
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
    return formatted_output

