#!/usr/bin/env python3
from ctransformers import AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncio
import uvicorn
import time
import os
import gc
import logging
from huggingface_hub import hf_hub_download

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="TinyLlama Inference Server")

# Global variables
model = None
model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
model_file = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"  # Using Q2 quantization for lower memory

# Model download with retry logic
def download_model_with_retry(model_name, model_file, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return hf_hub_download(repo_id=model_name, filename=model_file)
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Download failed, retrying... ({attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                logger.error(f"Failed to download model after {retries} attempts: {str(e)}")
                raise

class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1024)  # Reduced max length
    max_length: Optional[int] = Field(default=100, ge=1, le=1024)  # Reduced max length
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=2.0)

class BulkRequest(BaseModel):
    prompts: List[GenerateRequest]
    batch_size: int = Field(default=2, ge=1, le=10)  # Reduced batch size

@app.on_event("startup")
async def startup_event():
    global model
    try:
        # Download the model if not present
        logger.info(" Downloading model...")
        model_path = download_model_with_retry(model_name,model_file)
        
        # Initialize the model with memory-efficient settings
        logger.info(" Initializing model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama",
            context_length=1024,  # Reduced context length
            gpu_layers=0,  # CPU only for Mac compatibility
            batch_size=1,  # Process one at a time
            threads=2  # Limit thread usage
        )
        logger.info(" Model initialized successfully")
    except Exception as e:
        logger.info(f"Error initializing model: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {
        "status": "healthy",
        "model": model_name,
        "backend": "ctransformers"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text based on the prompt"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        start_time = time.time()
        
        # Generate text
        generated_text = model(
            request.text,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            stop=["</s>", "[/INST]"]
        )
        
        end_time = time.time()
        
        # Force garbage collection
        gc.collect()
        
        return {
            "text": generated_text,
            "usage": {
                "prompt_tokens": len(request.text.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(request.text.split()) + len(generated_text.split())
            },
            "generation_time": end_time - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_bulk")
async def generate_bulk(request: BulkRequest):
    """Generate text for multiple prompts"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        results = []
        errors = []
        start_time = time.time()
        
        # Process prompts in smaller batches
        for i in range(0, len(request.prompts), request.batch_size):
            batch = request.prompts[i:i + request.batch_size]
            
            # Process each prompt in the batch
            for prompt in batch:
                try:
                    generated_text = model(
                        prompt.text,
                        max_new_tokens=prompt.max_length,
                        temperature=prompt.temperature,
                        stop=["</s>", "[/INST]"]
                    )
                    
                    results.append({
                        "text": generated_text,
                        "error": None
                    })
                    
                    # Force garbage collection after each generation
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error generating text for prompt: {str(e)}")
                    results.append({
                        "text": None,
                        "error": str(e)
                    })
                    errors.append(str(e))
            
            # Small delay between batches to allow memory cleanup
            await asyncio.sleep(0.1)
        
        end_time = time.time()
        
        return {
            "results": results,
            "total_time": end_time - start_time,
            "successful": len([r for r in results if r["error"] is None]),
            "failed": len(errors),
            "errors": errors
        }
    except Exception as e:
        logger.error(f"Error in bulk generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        workers=1  # Single worker to reduce memory usage
    ) 