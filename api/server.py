from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from ctransformers import AutoModelForCausalLM
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import time
import asyncio
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE
from itertools import islice

# Initialize FastAPI app
app = FastAPI(
    title="TinyLlama API",
    description="API for running TinyLlama model inference",
    version="1.0.0"
)

# Model instance (loaded once and reused)
model = None
model_lock = asyncio.Lock()  # Lock for thread-safe model access

class PromptRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048, description="The input text to generate from")
    max_length: Optional[int] = Field(100, ge=1, le=2048, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")

class BulkRequest(BaseModel):
    prompts: List[PromptRequest] = Field(..., min_items=1, max_items=50, description="List of prompts to process")
    batch_size: Optional[int] = Field(5, ge=1, le=10, description="Number of concurrent requests")

class PromptResponse(BaseModel):
    prompt: str
    response: str
    time_taken: float
    tokens_generated: int
    status: str = "success"

class BulkResponse(BaseModel):
    results: List[PromptResponse]
    total_time: float
    successful: int
    failed: int
    errors: Optional[Dict[int, str]] = None

def load_model():
    """Load the model once and cache it."""
    global model
    if model is None:
        print("Loading model...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                model_type="llama",
                gpu_layers=0
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    return model

async def generate_text_safe(prompt: PromptRequest) -> PromptResponse:
    """Thread-safe text generation with error handling."""
    try:
        async with model_lock:  # Ensure thread-safe model access
            start_time = time.time()
            formatted_prompt = f"User: {prompt.text}\nAssistant:"
            
            response = model(
                formatted_prompt,
                max_new_tokens=prompt.max_length,
                temperature=prompt.temperature,
                stop=["</s>", "User:", "Assistant:"]
            )
            
            time_taken = time.time() - start_time
            tokens_generated = len(response.split())  # Approximate token count
            
            return PromptResponse(
                prompt=prompt.text,
                response=response,
                time_taken=time_taken,
                tokens_generated=tokens_generated
            )
    except Exception as e:
        return PromptResponse(
            prompt=prompt.text,
            response=str(e),
            time_taken=0.0,
            tokens_generated=0,
            status="error"
        )

@app.on_event("startup")
async def startup_event():
    """Load model when the server starts."""
    try:
        load_model()
    except Exception as e:
        print(f"Failed to load model on startup: {e}")

@app.post("/generate", response_model=PromptResponse)
async def generate_single(request: PromptRequest):
    """Generate text for a single prompt."""
    if model is None:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    return await generate_text_safe(request)

@app.post("/generate_bulk", response_model=BulkResponse)
async def generate_bulk(request: BulkRequest):
    """Generate text for multiple prompts in parallel using efficient batching."""
    if model is None:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )

    start_time = time.time()
    results = []
    errors = {}
    
    batch_size = request.batch_size  # Dynamically use batch size
    
    # Process in smaller chunks
    async def process_batch(batch):
        return await asyncio.gather(*[generate_text_safe(p) for p in batch])

    # Split into batches
    it = iter(request.prompts)
    while batch := list(islice(it, batch_size)):
        batch_results = await process_batch(batch)
        results.extend(batch_results)

    successful = sum(1 for r in results if r.status == "success")
    failed = len(results) - successful

    return BulkResponse(
        results=results,
        total_time=time.time() - start_time,
        successful=successful,
        failed=failed,
        errors={i: r.response for i, r in enumerate(results) if r.status == "error"}
    )

@app.get("/health")
async def health_check():
    """Check if the server and model are running."""
    return {
        "status": "healthy" if model is not None else "not_ready",
        "model_loaded": model is not None,
        "model_type": "TinyLlama-1.1B-Chat-v1.0-GGUF",
        "server_time": time.time()
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, workers=1) 