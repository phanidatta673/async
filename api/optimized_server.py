#!/usr/bin/env python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time
import asyncio
from ctransformers import AutoModelForCausalLM
import uvicorn
from concurrent.futures import ThreadPoolExecutor

# Initialize FastAPI app
app = FastAPI(title="Optimized Inference Server for Apple Silicon")

# Global variables
model = None
executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on your CPU cores

class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048)
    max_length: Optional[int] = Field(default=100, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=2.0)

class BulkRequest(BaseModel):
    prompts: List[GenerateRequest]
    batch_size: int = Field(default=5, ge=1, le=50)

@app.on_event("startup")
async def startup_event():
    global model
    try:
        # Initialize model with Metal acceleration
        print("🚀 Initializing model with Metal acceleration...")
        model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            model_type="llama",
            gpu_layers=50,  # Adjust based on your memory
            context_length=2048,
            batch_size=8,
            threads=8  # Adjust based on your CPU cores
        )
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing model: {str(e)}")
        raise

def generate_text(prompt: str, max_length: int, temperature: float) -> str:
    """Generate text using the model in a thread-safe way"""
    try:
        return model(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            stop=["</s>", "[/INST]"]
        )
    except Exception as e:
        raise Exception(f"Generation error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {
        "status": "healthy",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "backend": "ctransformers with Metal acceleration"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text based on the prompt"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        generated_text = await loop.run_in_executor(
            executor,
            generate_text,
            request.text,
            request.max_length,
            request.temperature
        )
        
        end_time = time.time()
        
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
        loop = asyncio.get_event_loop()
        
        # Process prompts in batches
        for i in range(0, len(request.prompts), request.batch_size):
            batch = request.prompts[i:i + request.batch_size]
            batch_tasks = []
            
            # Create tasks for each prompt in the batch
            for prompt in batch:
                task = loop.run_in_executor(
                    executor,
                    generate_text,
                    prompt.text,
                    prompt.max_length,
                    prompt.temperature
                )
                batch_tasks.append(task)
            
            # Wait for all tasks in the batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({"text": None, "error": str(result)})
                    errors.append(str(result))
                else:
                    results.append({"text": result, "error": None})
        
        end_time = time.time()
        
        return {
            "results": results,
            "total_time": end_time - start_time,
            "successful": len([r for r in results if r["error"] is None]),
            "failed": len(errors),
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🍎 Starting optimized server for Apple Silicon...")
    uvicorn.run(app, host="0.0.0.0", port=8001) 