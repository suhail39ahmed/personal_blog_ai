# main.py

import gradio as gr
from utils import load_prompt, generate_general_response

# Load custom responses
about_me_response = load_prompt("prompts/about_me.txt")
linkedin_response = load_prompt("prompts/linkedin.txt")

def chatbot_response(input_text):
    input_text_lower = input_text.lower()
    
    # Check for specific questions about you
    if "about you" in input_text_lower or "your experience" in input_text_lower:
        return about_me_response
    elif "linkedin" in input_text_lower:
        return linkedin_response
    else:
        # For general questions, use the model to generate a response
        return generate_general_response(input_text)

# Create a Gradio interface
iface = gr.Interface(
    fn=chatbot_response, 
    inputs="text", 
    outputs="text", 
    title="Personal Blog Assistant",
    description="Ask me about my background, LinkedIn profile, or general knowledge questions."
)

if __name__ == "__main__":
    iface.launch()
