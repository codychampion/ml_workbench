#!/usr/bin/env python3
"""
ComfyUI Mock Server (Phase 1)
=============================
A minimal HTTP server that simulates ComfyUI's API for Phase 1 development.
Allows testing the pipeline integration without the full ComfyUI installation.

PHASE 2/3 TODO: Replace with actual ComfyUI installation
"""

import json
import os
import uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image


class ComfyUIConfig:
    """Configuration for the mock ComfyUI server."""
    HOST = os.getenv("COMFYUI_HOST", "0.0.0.0")
    PORT = int(os.getenv("COMFYUI_PORT", "8188"))
    OUTPUT_DIR = Path(os.getenv("COMFYUI_OUTPUT_DIR", "/workspace/comfyui/output"))
    INPUT_DIR = Path(os.getenv("COMFYUI_INPUT_DIR", "/workspace/comfyui/input"))
    MODELS_DIR = Path(os.getenv("COMFYUI_MODELS_DIR", "/workspace/comfyui/models"))


# Ensure directories exist
ComfyUIConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ComfyUIConfig.INPUT_DIR.mkdir(parents=True, exist_ok=True)
ComfyUIConfig.MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Job queue simulation
JOBS: Dict[str, Dict[str, Any]] = {}


class ComfyUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler simulating ComfyUI API."""

    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error(self, message: str, status: int = 400):
        """Send error response."""
        self._send_json({"error": message}, status)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self._send_json({"status": "healthy", "service": "comfyui-mock"})

        elif self.path == "/":
            self._send_json({
                "service": "ComfyUI Mock Server (Phase 1)",
                "version": "1.0.0",
                "phase": "1-local-core",
                "endpoints": [
                    "GET /health - Health check",
                    "GET /system_stats - System statistics",
                    "GET /object_info - Available nodes",
                    "POST /prompt - Queue a generation job",
                    "GET /history/<job_id> - Get job history",
                ],
                "note": "PHASE 2/3 TODO: Replace with actual ComfyUI"
            })

        elif self.path == "/system_stats":
            self._send_json({
                "system": {
                    "os": "linux",
                    "python_version": "3.11",
                    "pytorch_version": "2.0+",
                    "device": "cpu",  # Phase 1: CPU only
                },
                "devices": [
                    {"name": "cpu", "type": "cpu", "vram_total": 0, "vram_free": 0}
                ],
                # PHASE 2/3 TODO: Add GPU stats
                # {"name": "cuda:0", "type": "cuda", "vram_total": 24576, "vram_free": 20000}
            })

        elif self.path == "/object_info":
            # Simplified node info for Phase 1
            self._send_json({
                "KSampler": {
                    "input": {"required": {"model": ["MODEL"], "seed": ["INT"]}},
                    "output": ["LATENT"],
                    "category": "sampling"
                },
                "CheckpointLoader": {
                    "input": {"required": {"ckpt_name": ["STRING"]}},
                    "output": ["MODEL", "CLIP", "VAE"],
                    "category": "loaders"
                },
                "CLIPTextEncode": {
                    "input": {"required": {"text": ["STRING"], "clip": ["CLIP"]}},
                    "output": ["CONDITIONING"],
                    "category": "conditioning"
                },
                "VAEDecode": {
                    "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
                    "output": ["IMAGE"],
                    "category": "latent"
                },
                "SaveImage": {
                    "input": {"required": {"images": ["IMAGE"]}},
                    "output": [],
                    "category": "image"
                }
            })

        elif self.path.startswith("/history/"):
            job_id = self.path.split("/")[-1]
            if job_id in JOBS:
                self._send_json({job_id: JOBS[job_id]})
            else:
                self._send_error(f"Job {job_id} not found", 404)

        elif self.path == "/history":
            self._send_json(JOBS)

        else:
            self._send_error(f"Unknown endpoint: {self.path}", 404)

    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode() if content_length > 0 else "{}"

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_error("Invalid JSON")
            return

        if self.path == "/prompt":
            # Queue a generation job
            job_id = str(uuid.uuid4())
            prompt = data.get("prompt", {})

            # Simulate job execution (Phase 1: generate placeholder image)
            output_images = self._generate_placeholder_images(job_id, prompt)

            JOBS[job_id] = {
                "status": "completed",
                "prompt": prompt,
                "outputs": {"images": output_images},
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
            }

            self._send_json({
                "prompt_id": job_id,
                "status": "queued",
                "message": "Phase 1: Job executed synchronously (mock)"
            })

        elif self.path == "/interrupt":
            self._send_json({"status": "interrupted"})

        else:
            self._send_error(f"Unknown endpoint: {self.path}", 404)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _generate_placeholder_images(self, job_id: str, prompt: dict) -> list:
        """Generate placeholder images for Phase 1 testing."""
        # Create a simple gradient image
        width, height = 512, 512

        # Generate colorful noise pattern
        np.random.seed(hash(job_id) % (2**32))
        img_array = np.random.rand(height, width, 3) * 0.3 + 0.35

        # Add gradient
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        img_array[:, :, 0] += xx * 0.3
        img_array[:, :, 1] += yy * 0.3
        img_array[:, :, 2] += (xx + yy) * 0.15

        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)

        # Save image
        output_filename = f"comfyui_{job_id[:8]}.png"
        output_path = ComfyUIConfig.OUTPUT_DIR / output_filename
        Image.fromarray(img_array).save(output_path)

        print(f"[ComfyUI Mock] Generated: {output_path}")

        return [{
            "filename": output_filename,
            "subfolder": "",
            "type": "output"
        }]

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[ComfyUI] {self.address_string()} - {format % args}")


def main():
    """Start the ComfyUI mock server."""
    server = HTTPServer((ComfyUIConfig.HOST, ComfyUIConfig.PORT), ComfyUIHandler)

    print("=" * 60)
    print("ComfyUI Mock Server (Phase 1)")
    print("=" * 60)
    print(f"Host: {ComfyUIConfig.HOST}")
    print(f"Port: {ComfyUIConfig.PORT}")
    print(f"Output: {ComfyUIConfig.OUTPUT_DIR}")
    print("-" * 60)
    print("PHASE 2/3 TODO: Replace with actual ComfyUI installation")
    print("=" * 60)
    print(f"\nServer running at http://{ComfyUIConfig.HOST}:{ComfyUIConfig.PORT}")
    print("Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
