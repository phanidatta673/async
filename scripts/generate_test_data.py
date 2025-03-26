#!/usr/bin/env python
import json
import random
import os
from itertools import product

def generate_large_test_data(num_examples: int = 1000):
    """Generate a large test data file with varied prompts."""
    
    # Define a rich set of templates and topics
    templates = [
        "Explain {topic} in simple terms",
        "What are the key concepts of {topic}?",
        "How does {topic} work?",
        "What are the applications of {topic}?",
        "Compare and contrast different approaches to {topic}",
        "What are the advantages and disadvantages of {topic}?",
        "Describe the history and evolution of {topic}",
        "What are the best practices for implementing {topic}?",
        "How can {topic} be optimized for better performance?",
        "What are the future trends in {topic}?"
    ]
    
    topics = [
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "neural networks",
        "data science",
        "big data analytics",
        "cloud computing",
        "edge computing",
        "distributed systems",
        "blockchain technology",
        "cybersecurity",
        "quantum computing"
    ]
    
    # Generate varied prompts
    prompts = []
    combinations = list(product(templates, topics))
    
    # First use all unique combinations
    for template, topic in combinations:
        if len(prompts) >= num_examples:
            break
        prompt = {
            "text": template.format(topic=topic),
            "max_length": random.randint(50, 200),
            "temperature": round(random.uniform(0.5, 0.9), 2)
        }
        prompts.append(prompt)
    
    # If we need more, generate additional variations with different parameters
    while len(prompts) < num_examples:
        template = random.choice(templates)
        topic = random.choice(topics)
        prompt = {
            "text": template.format(topic=topic),
            "max_length": random.randint(50, 200),
            "temperature": round(random.uniform(0.5, 0.9), 2)
        }
        prompts.append(prompt)
    
    # Ensure output directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save the test data
    output_file = "data/test_data.json"
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Generated {len(prompts)} prompts with {len(set(p['text'] for p in prompts))} unique combinations")
    print(f"Test data saved to {output_file}")

if __name__ == "__main__":
    generate_large_test_data(1000) 