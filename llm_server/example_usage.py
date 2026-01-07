#!/usr/bin/env python3
"""Example usage of local LLM server for ML workbench tasks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
import json

# Initialize client
llm = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key-here"
)


def test_basic_chat():
    """Test basic chat completion."""
    print("=== Basic Chat ===")

    response = llm.chat.completions.create(
        model="meta-llama/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "user", "content": "Explain adversarial ML in one sentence."}
        ]
    )

    print(response.choices[0].message.content)
    print()


def test_streaming():
    """Test streaming response."""
    print("=== Streaming ===")

    stream = llm.chat.completions.create(
        model="meta-llama/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "user", "content": "Write a haiku about neural networks."}
        ],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def generate_attack_prompts(model_type: str) -> list:
    """Generate adversarial attack prompts using LLM."""
    print(f"=== Generating Attack Prompts for {model_type} ===")

    prompt = f"""You are a red-team AI security expert. Generate 5 creative adversarial test cases for a {model_type} model.

For each test case, provide:
1. Attack vector
2. Expected behavior
3. Success criteria

Format as JSON array."""

    response = llm.chat.completions.create(
        model="meta-llama/Llama-3.1-70B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    print(response.choices[0].message.content)
    print()
    return response.choices[0].message.content


def analyze_vulnerability_report(vulnerabilities: dict) -> str:
    """Analyze ML vulnerability report and suggest mitigations."""
    print("=== Analyzing Vulnerability Report ===")

    prompt = f"""You are an ML security consultant. Analyze this vulnerability report and provide:
1. Severity ranking
2. Root cause analysis
3. Mitigation strategies
4. Priority recommendations

Report:
{json.dumps(vulnerabilities, indent=2)}

Provide detailed, actionable advice."""

    response = llm.chat.completions.create(
        model="meta-llama/Llama-3.1-70B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500
    )

    analysis = response.choices[0].message.content
    print(analysis)
    print()
    return analysis


def code_review_assistant(code: str, language: str = "python") -> str:
    """Review code for security issues."""
    print(f"=== Code Review ({language}) ===")

    prompt = f"""Review this {language} code for security vulnerabilities, especially:
- SQL injection
- Command injection
- Path traversal
- Insecure deserialization
- Hardcoded credentials
- ML-specific attacks (adversarial inputs, model extraction)

Code:
```{language}
{code}
```

Provide specific issues and fixes."""

    response = llm.chat.completions.create(
        model="meta-llama/Llama-3.1-70B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    review = response.choices[0].message.content
    print(review)
    print()
    return review


def dataset_metadata_generator(dataset_path: str, num_samples: int) -> dict:
    """Generate dataset documentation using LLM."""
    print("=== Dataset Metadata Generation ===")

    prompt = f"""Create comprehensive metadata for a dataset with these properties:
- Path: {dataset_path}
- Samples: {num_samples}

Generate:
1. Dataset name
2. Description
3. Potential use cases
4. Data quality concerns
5. Ethical considerations
6. Suggested preprocessing steps

Format as JSON."""

    response = llm.chat.completions.create(
        model="meta-llama/Llama-3.1-70B-Instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    print(response.choices[0].message.content)
    print()
    return response.choices[0].message.content


def vision_analysis(image_path: str):
    """Analyze image using vision model (if running on port 8002)."""
    print(f"=== Vision Analysis: {image_path} ===")

    vision_client = OpenAI(
        base_url="http://localhost:8002/v1",
        api_key="your-secret-key-here"
    )

    response = vision_client.chat.completions.create(
        model="Qwen/Qwen2-VL-7B-Instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail. Identify any potential security concerns or adversarial patterns."},
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
            ]
        }]
    )

    print(response.choices[0].message.content)
    print()


def main():
    """Run all examples."""

    # Basic tests
    test_basic_chat()
    test_streaming()

    # ML security tasks
    generate_attack_prompts("image classifier")

    # Vulnerability analysis
    sample_vulns = {
        "model": "facial_recognition_v1",
        "vulnerabilities": [
            {"type": "adversarial_evasion", "severity": "high", "success_rate": 0.87},
            {"type": "model_extraction", "severity": "critical", "queries": 1000}
        ]
    }
    analyze_vulnerability_report(sample_vulns)

    # Code review
    sample_code = """
def classify_image(image_path):
    model = load_model('classifier.pth')
    img = Image.open(image_path)
    result = model(img)
    return result
"""
    code_review_assistant(sample_code)

    # Dataset documentation
    dataset_metadata_generator("/data/custom_dataset", 10000)

    print("All examples completed!")


if __name__ == "__main__":
    main()
