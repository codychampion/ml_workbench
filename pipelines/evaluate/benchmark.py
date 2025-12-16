#!/usr/bin/env python3
"""
Model Benchmarking
==================
Run benchmarks on trained models to measure performance.

Usage:
    python -m pipelines.evaluate.benchmark --model ./models/captioner/v1 --dataset ./data/test
    python -m pipelines.evaluate.benchmark --model blip-base --benchmark captioning
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm


class ModelBenchmark:
    """Benchmark model performance."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._get_device(device)
        self.results = {}

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def benchmark_latency(
        self,
        input_data,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> dict:
        """Measure inference latency."""
        latencies = []

        # Warmup
        for _ in range(warmup_runs):
            self._run_inference(input_data)

        # Benchmark
        for _ in tqdm(range(num_runs), desc="Benchmarking latency"):
            start = time.perf_counter()
            self._run_inference(input_data)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        return {
            "mean_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "p50_ms": sorted(latencies)[len(latencies) // 2],
            "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "num_runs": num_runs
        }

    def benchmark_throughput(
        self,
        input_data,
        duration_seconds: float = 30.0
    ) -> dict:
        """Measure inference throughput."""
        count = 0
        start = time.perf_counter()

        while time.perf_counter() - start < duration_seconds:
            self._run_inference(input_data)
            count += 1

        elapsed = time.perf_counter() - start
        return {
            "samples_per_second": count / elapsed,
            "total_samples": count,
            "duration_seconds": elapsed
        }

    def benchmark_memory(self) -> dict:
        """Measure memory usage."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            # Run a sample inference
            self._run_inference(self._get_sample_input())
            return {
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                "current_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024
            }
        return {"note": "Memory benchmarking only available on CUDA"}

    def _run_inference(self, input_data):
        """Override in subclass."""
        pass

    def _get_sample_input(self):
        """Override in subclass."""
        pass

    def save_results(self, output_path: Path):
        """Save benchmark results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "model": self.model_path,
            "device": self.device,
            "timestamp": datetime.now().isoformat(),
            "benchmarks": self.results
        }

        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {output_path}")
        return results


class CaptionerBenchmark(ModelBenchmark):
    """Benchmark captioning models."""

    def __init__(self, model_path: str, device: str = "auto"):
        super().__init__(model_path, device)
        self._load_model()

    def _load_model(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration

        print(f"Loading model: {self.model_path}")
        self.processor = BlipProcessor.from_pretrained(self.model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_path
        ).to(self.device)
        self.model.eval()

    def _get_sample_input(self):
        # Create a sample image
        return Image.new("RGB", (512, 512), color="white")

    def _run_inference(self, input_data):
        image = input_data if input_data else self._get_sample_input()
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=50)
        return self.processor.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ML models")

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--benchmark",
        choices=["captioning", "all"],
        default="captioning",
        help="Benchmark type"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./outputs/benchmarks"),
        help="Output directory"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of runs for latency benchmark"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto"
    )

    args = parser.parse_args()

    if args.benchmark in ["captioning", "all"]:
        bench = CaptionerBenchmark(args.model, args.device)

        print("\n" + "=" * 60)
        print("Running Captioner Benchmarks")
        print("=" * 60)

        # Latency
        print("\n[1/3] Latency benchmark...")
        bench.results["latency"] = bench.benchmark_latency(
            bench._get_sample_input(),
            num_runs=args.num_runs
        )

        # Memory
        print("\n[2/3] Memory benchmark...")
        bench.results["memory"] = bench.benchmark_memory()

        # Throughput
        print("\n[3/3] Throughput benchmark (30s)...")
        bench.results["throughput"] = bench.benchmark_throughput(
            bench._get_sample_input(),
            duration_seconds=30
        )

        # Save results
        output_file = args.output / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results = bench.save_results(output_file)

        # Print summary
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Device: {bench.device}")
        print(f"\nLatency:")
        print(f"  Mean: {results['benchmarks']['latency']['mean_ms']:.2f} ms")
        print(f"  P95:  {results['benchmarks']['latency']['p95_ms']:.2f} ms")
        print(f"\nThroughput:")
        print(f"  {results['benchmarks']['throughput']['samples_per_second']:.2f} samples/sec")


if __name__ == "__main__":
    main()
