#!/usr/bin/env python
import json
import time
import asyncio
import aiohttp
from datetime import datetime
import os

class DockerInferenceTester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.test_data_file = "data/test_data.json"
        self.results_dir = "results/docker_inference"
        os.makedirs(self.results_dir, exist_ok=True)

    async def check_server_health(self):
        """Check if the Docker inference server is healthy"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        print(f"Server Status: {health_data.get('status')}")
                        print(f"Model: {health_data.get('model')}")
                        print(f"Backend: {health_data.get('backend')}")
                        return True
                    else:
                        print(f"Server returned status {response.status}")
                        return False
            except Exception as e:
                print(f"Error connecting to server: {str(e)}")
                return False

    async def test_single_inference(self, prompt: str) -> dict:
        """Test a single inference request"""
        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": str(prompt),  # Ensure prompt is a string
                        "max_length": 100,
                        "temperature": 0.7
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()
                    end_time = time.time()
                    return {
                        "prompt": prompt,
                        "response": result,
                        "time_taken": end_time - start_time,
                        "status": response.status
                    }
            except Exception as e:
                return {
                    "prompt": prompt,
                    "error": str(e),
                    "time_taken": time.time() - start_time,
                    "status": getattr(e, 'status', 500)
                }

    async def test_bulk_inference(self, prompts: list, batch_size: int = 5) -> dict:
        """Test bulk inference requests"""
        async with aiohttp.ClientSession() as session:
            # Format prompts according to the expected schema
            formatted_prompts = [
                {
                    "text": str(prompt),  # Ensure prompt is a string
                    "max_length": 100,
                    "temperature": 0.7
                }
                for prompt in prompts
            ]
            
            payload = {
                "prompts": formatted_prompts,
                "batch_size": batch_size
            }
            
            try:
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/generate_bulk",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)  # Increased timeout for bulk requests
                ) as response:
                    response_text = await response.text()
                    try:
                        result = json.loads(response_text)
                    except json.JSONDecodeError:
                        result = {"error": f"Invalid JSON response: {response_text[:200]}..."}
                    
                    end_time = time.time()
                    return {
                        "num_prompts": len(prompts),
                        "batch_size": batch_size,
                        "response": result,
                        "total_time": end_time - start_time,
                        "status": response.status
                    }
            except Exception as e:
                return {
                    "num_prompts": len(prompts),
                    "batch_size": batch_size,
                    "error": str(e),
                    "total_time": time.time() - start_time,
                    "status": getattr(e, 'status', 500)
                }

    def save_results(self, results: dict, test_type: str):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/{test_type}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")

async def main():
    tester = DockerInferenceTester()
    
    # Check server health
    print("\nChecking Docker inference server health...")
    if not await tester.check_server_health():
        print("Server health check failed. Please ensure the Docker container is running.")
        return

    # Load test data
    try:
        with open("data/test_data.json", 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("Test data file not found. Please ensure data/test_data.json exists.")
        return

    # # Test single requests
    # print("\nTesting single requests...")
    # single_results = []
    # for i, prompt in enumerate(test_data[:5]):  # Test first 5 prompts
    #     print(f"Processing single request {i+1}/5...")
    #     result = await tester.test_single_inference(prompt)
    #     single_results.append(result)
    #     if result.get('status') != 200:
    #         print(f"Warning: Request failed with status {result.get('status')}")
    #         print(f"Error: {result.get('error') or result.get('response')}")
    # tester.save_results(single_results, "single_requests")

    # Test bulk requests
    print("\nTesting bulk requests...")
    bulk_results = []
    batch_size = 5  # Fixed batch size
    
    # Process prompts in chunks of 5
    for i in range(0, min(len(test_data), 5), batch_size):
        chunk = test_data[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} (size: {len(chunk)})")
        result = await tester.test_bulk_inference(chunk, batch_size)
        if result.get('status') != 200:
            print(f"Warning: Bulk request failed with status {result.get('status')}")
            error_msg = result.get('error') or result.get('response', {}).get('detail', 'Unknown error')
            print(f"Error: {error_msg}")
        bulk_results.append(result)
    
    tester.save_results(bulk_results, "bulk_requests")

    # Print summary
    print("\nTest Summary:")
    print("Single Requests:")
    # successful_singles = [r for r in single_results if r.get('status') == 200]
    # print(f"- Successful: {len(successful_singles)}/{len(single_results)}")
    # if successful_singles:
    #     avg_time = sum(r['time_taken'] for r in successful_singles) / len(successful_singles)
    #     print(f"- Average response time: {avg_time:.2f}s")

    print("\nBulk Requests:")
    successful_bulks = [r for r in bulk_results if r.get('status') == 200]
    print(f"- Successful batches: {len(successful_bulks)}/{len(bulk_results)}")
    if successful_bulks:
        total_prompts = sum(r['num_prompts'] for r in successful_bulks)
        total_time = sum(r['total_time'] for r in successful_bulks)
        print(f"- Total prompts processed: {total_prompts}")
        print(f"- Total processing time: {total_time:.2f}s")
        print(f"- Average time per prompt: {total_time/total_prompts:.2f}s")
    else:
        print("- No successful bulk requests")

if __name__ == "__main__":
    asyncio.run(main()) 