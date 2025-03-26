#!/usr/bin/env python
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import CompletionRequest
from vllm.utils import random_uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uvicorn
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize FastAPI app
app = FastAPI(title="vLLM Inference Server")

# Global variables
engine = None

class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048)
    max_length: Optional[int] = Field(default=100, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=2.0)

class BulkRequest(BaseModel):
    prompts: List[GenerateRequest]
    batch_size: int = Field(default=5, ge=1, le=50)

@app.on_event("startup")
async def startup_event():
    global engine
    
    # Configure engine arguments
    engine_args = AsyncEngineArgs(
        model="TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",  # Use float16 for better performance
        gpu_memory_utilization=0.90,  # Use 90% of GPU memory
        max_num_batched_tokens=4096,  # Adjust based on your GPU memory
        trust_remote_code=True,
        tensor_parallel_size=1  # Adjust if using multiple GPUs
    )
    
    # Initialize the engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("✅ vLLM engine initialized successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {"status": "healthy", "model": "TinyLlama-1.1B-Chat-v1.0"}

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text based on the prompt"""
    try:
        # Create sampling parameters
        sampling_params = {
            "max_tokens": request.max_length,
            "temperature": request.temperature
        }
        
        # Generate text
        results_generator = engine.generate(request.text, sampling_params, request_id=random_uuid())
        
        # Wait for results
        final_output = None
        async for output in results_generator:
            final_output = output
        
        if final_output is None:
            raise HTTPException(status_code=500, detail="No output generated")
            
        # Get the generated text from the first sequence
        generated_text = final_output.outputs[0].text
        
        return {
            "text": generated_text,
            "usage": {
                "prompt_tokens": len(request.text.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(request.text.split()) + len(generated_text.split())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_bulk")
async def generate_bulk(request: BulkRequest):
    """Generate text for multiple prompts"""
    try:
        results = []
        errors = []
        start_time = asyncio.get_event_loop().time()
        
        # Process prompts in batches
        for i in range(0, len(request.prompts), request.batch_size):
            batch = request.prompts[i:i + request.batch_size]
            batch_results = []
            
            # Process each prompt in the batch
            for prompt in batch:
                try:
                    sampling_params = {
                        "max_tokens": prompt.max_length,
                        "temperature": prompt.temperature
                    }
                    
                    results_generator = engine.generate(prompt.text, sampling_params, request_id=random_uuid())
                    final_output = None
                    async for output in results_generator:
                        final_output = output
                    
                    if final_output is None:
                        raise Exception("No output generated")
                        
                    generated_text = final_output.outputs[0].text
                    batch_results.append({
                        "text": generated_text,
                        "error": None
                    })
                except Exception as e:
                    batch_results.append({
                        "text": None,
                        "error": str(e)
                    })
                    errors.append(str(e))
            
            results.extend(batch_results)
        
        end_time = asyncio.get_event_loop().time()
        
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
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  No GPU detected. vLLM performs best with GPU acceleration.")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8001) 