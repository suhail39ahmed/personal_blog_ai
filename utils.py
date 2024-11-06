# utils.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer globally
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def load_prompt(file_path):
    """Load a custom response from a file."""
    with open(file_path, "r") as file:
        return file.read().strip()

def generate_general_response(input_text):
    """Generate a general knowledge response using the language model."""
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
