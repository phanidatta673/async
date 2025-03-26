#!/usr/bin/env python
from ctransformers import AutoModelForCausalLM

def load_model():
    """Load the quantized model."""
    print("Loading model...")
    
    # Load 4-bit quantized model
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",  # Updated correct model file name
        model_type="llama",
        gpu_layers=0  # Use CPU only for stable performance on 8GB RAM
    )
    
    return model

def generate_text(prompt, model, max_length=100, temperature=0.7):
    """Generate text from a prompt."""
    response = model(
        prompt,
        max_new_tokens=max_length,
        temperature=temperature,
        stop=["</s>", "User:", "Assistant:"]
    )
    return response

def main():
    # Load model
    model = load_model()
    
    print("\nTinyLlama-1.1B Chat Model (4-bit quantized) loaded! Type 'quit' to exit.")
    print("Enter your prompt:")
    
    while True:
        prompt = input("\n> ")
        if prompt.lower() == 'quit':
            break
        
        # Format prompt for chat
        formatted_prompt = f"User: {prompt}\nAssistant:"
        
        try:
            response = generate_text(formatted_prompt, model)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main() 