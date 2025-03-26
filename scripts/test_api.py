#!/usr/bin/env python
import json
import time
import asyncio
import aiohttp
import random
import pandas as pd
from datetime import datetime
from typing import List, Dict
import os

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.batch_results = []
        self.test_data_file = "data/test_data.json"

    def generate_test_data(self, sample_file: str, num_requests: int = 1) -> List[Dict]:
        """Load and sample from pre-generated test data."""
        try:
            with open(self.test_data_file, 'r') as f:
                all_data = json.load(f)
                
            if num_requests >= len(all_data):
                print(f"Requested {num_requests} samples but only {len(all_data)} available. Using all available data.")
                return all_data
            
            # Randomly sample num_requests prompts
            sampled_data = random.sample(all_data, num_requests)
            print(f"Sampled {len(sampled_data)} prompts from {len(all_data)} available prompts.")
            return sampled_data
                
        except FileNotFoundError:
            print("No test data found. Please run generate_test_data.py first to create test data.")
            print("Command: python scripts/generate_test_data.py")
            return []

    async def test_single_request(self, prompt: Dict) -> Dict:
        """Test a single prompt request."""
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.base_url}/generate",
                    json=prompt,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()
                    end_time = time.time()
                    result['request_time'] = end_time - start_time
                    return result
            except Exception as e:
                error_msg = f"Error in request: {str(e)}"
                print(error_msg)
                return {
                    "error": error_msg,
                    "request_time": time.time() - start_time
                }

    async def test_bulk_request(self, prompts: List[Dict], batch_size: int = 5) -> Dict:
        """Test a bulk request."""
        if not prompts:  # Skip if no prompts
            return {"error": "No prompts to process", "total_request_time": 0}

        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.base_url}/generate_bulk",
                    json={"prompts": prompts, "batch_size": batch_size},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    result = await response.json()
                    end_time = time.time()
                    result['total_request_time'] = end_time - start_time
                    return result
            except Exception as e:
                error_msg = f"Error in bulk request: {str(e)}"
                print(error_msg)
                return {
                    "error": error_msg,
                    "total_request_time": time.time() - start_time
                }

    def save_results(self, output_dir: str = "results"):
        """Save test results to CSV files."""
        if not (self.results or self.batch_results):  # Skip if no results
            print("No results to save.")
            return

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.results:
            df_single = pd.DataFrame(self.results)
            df_single.to_csv(f"{output_dir}/single_requests_{timestamp}.csv", index=False)
        
        if self.batch_results:
            df_bulk = pd.DataFrame(self.batch_results)
            df_bulk.to_csv(f"{output_dir}/bulk_requests_{timestamp}.csv", index=False)

    async def run_tests(self, num_requests: int = 1, batch_sizes: List[int] = [5]):
        """Run complete test suite."""
        print("Starting test suite...")
        prompts = self.generate_test_data("data/sample_prompts.json", num_requests)
        
        if not prompts:
            print("No test data available. Exiting...")
            return

        # Test single requests
        print(f"\nTesting {len(prompts)} single requests...")
        tasks = [self.test_single_request(prompt) for prompt in prompts]
        self.results = await asyncio.gather(*tasks)
        
        # For num_requests=1, skip bulk testing
        if num_requests == 1:
            print("\nSkipping bulk requests for single request test.")
        else:
            # Test bulk requests with different batch sizes
            print("\nTesting bulk requests with different batch sizes...")
            for batch_size in batch_sizes:
                print(f"\nBatch size: {batch_size}")
                # Only process if we have prompts
                if prompts:
                    result = await self.test_bulk_request(prompts, batch_size)
                    self.batch_results.append({
                        "batch_size": batch_size,
                        "num_prompts": len(prompts),
                        **result
                    })
        
        # Save results
        self.save_results()
        
        # Print summary
        print("\nTest Summary:")
        print(f"Single Requests ({len(self.results)} requests):")
        if self.results:
            successful_results = [r for r in self.results if 'error' not in r]
            if successful_results:
                avg_time = sum(r['request_time'] for r in successful_results) / len(successful_results)
                print(f"- Average response time: {avg_time:.2f}s")
            else:
                print("- No successful requests")
        
        if num_requests > 1:
            print(f"\nBulk Requests:")
            for batch_size in batch_sizes:
                batch_data = [r for r in self.batch_results if r['batch_size'] == batch_size]
                if batch_data:
                    successful_batches = [r for r in batch_data if 'error' not in r]
                    if successful_batches:
                        avg_time = sum(r['total_request_time'] for r in successful_batches) / len(successful_batches)
                        print(f"- Batch size {batch_size}: Average time per batch: {avg_time:.2f}s")
                    else:
                        print(f"- Batch size {batch_size}: No successful requests")

async def main():
    tester = APITester()
    await tester.run_tests(num_requests=1)

if __name__ == "__main__":
    asyncio.run(main()) 