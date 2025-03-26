#!/usr/bin/env python
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    """
    Downloads the TinyLlama-1.1B model from Hugging Face.
    Uses local cache if the model is already downloaded.
    """
    print("Downloading TinyLlama-1.1B model from Hugging Face...")
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained("models/tinyllama-1.1b")
        
        # Download model
        print("Downloading model (this may take a while)...")
        # Use float32 for macOS compatibility
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float32
        )
        model.save_pretrained("models/tinyllama-1.1b")
        
        print("Model and tokenizer downloaded successfully to models/tinyllama-1.1b")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nNOTE: If you encounter any issues, please check:")
        print("1. Your internet connection")
        print("2. Available disk space")

if __name__ == "__main__":
    download_model() 