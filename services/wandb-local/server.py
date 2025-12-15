#!/usr/bin/env python3
"""
W&B Local Server (Phase 1)
==========================
Provides utilities for managing offline W&B runs.
In Phase 1, we use offline mode exclusively.

PHASE 2/3 TODO:
- Deploy wandb/local server for self-hosted tracking
- Or configure for W&B cloud with API keys
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler


class WandBLocalConfig:
    """W&B local server configuration."""
    HOST = os.getenv("WANDB_LOCAL_HOST", "0.0.0.0")
    PORT = int(os.getenv("WANDB_LOCAL_PORT", "8080"))
    RUNS_DIR = Path(os.getenv("WANDB_DIR", "/workspace/wandb/runs"))
    ARTIFACTS_DIR = Path(os.getenv("WANDB_ARTIFACTS_DIR", "/workspace/wandb/artifacts"))


# Ensure directories exist
WandBLocalConfig.RUNS_DIR.mkdir(parents=True, exist_ok=True)
WandBLocalConfig.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class WandBLocalHandler(BaseHTTPRequestHandler):
    """HTTP handler for W&B local utilities."""

    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/" or self.path == "/health":
            self._send_json({
                "service": "W&B Local Utilities (Phase 1)",
                "status": "healthy",
                "mode": "offline",
                "runs_dir": str(WandBLocalConfig.RUNS_DIR),
                "endpoints": [
                    "GET /health - Health check",
                    "GET /runs - List offline runs",
                    "GET /runs/<run_id> - Get run details",
                    "POST /sync - Sync offline runs (Phase 2/3)",
                ],
                "phase_1_note": "Using wandb offline mode. Sync when online with: wandb sync <run_dir>"
            })

        elif self.path == "/runs":
            runs = self._list_offline_runs()
            self._send_json({
                "count": len(runs),
                "runs": runs,
                "sync_command": f"wandb sync {WandBLocalConfig.RUNS_DIR}/offline-*"
            })

        elif self.path.startswith("/runs/"):
            run_id = self.path.split("/")[-1]
            run_info = self._get_run_info(run_id)
            if run_info:
                self._send_json(run_info)
            else:
                self._send_json({"error": f"Run {run_id} not found"}, 404)

        else:
            self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/sync":
            # PHASE 2/3 TODO: Implement actual sync
            self._send_json({
                "status": "not_implemented",
                "message": "Phase 1: Use 'wandb sync' CLI command when online",
                "command": f"wandb sync {WandBLocalConfig.RUNS_DIR}/offline-*"
            })
        else:
            self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

    def _list_offline_runs(self) -> list:
        """List all offline W&B runs."""
        runs = []

        for run_dir in WandBLocalConfig.RUNS_DIR.glob("offline-*"):
            if run_dir.is_dir():
                run_info = {
                    "id": run_dir.name,
                    "path": str(run_dir),
                    "created": datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat(),
                    "size_mb": sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file()) / (1024 * 1024)
                }

                # Try to read run metadata
                config_file = run_dir / "files" / "config.yaml"
                if config_file.exists():
                    run_info["has_config"] = True

                runs.append(run_info)

        # Sort by creation time (newest first)
        runs.sort(key=lambda x: x["created"], reverse=True)
        return runs

    def _get_run_info(self, run_id: str) -> dict:
        """Get detailed info for a specific run."""
        run_dir = WandBLocalConfig.RUNS_DIR / run_id

        if not run_dir.exists():
            # Try with offline- prefix
            run_dir = WandBLocalConfig.RUNS_DIR / f"offline-{run_id}"

        if not run_dir.exists():
            return None

        files = list(run_dir.rglob("*"))
        file_list = [str(f.relative_to(run_dir)) for f in files if f.is_file()]

        return {
            "id": run_dir.name,
            "path": str(run_dir),
            "created": datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat(),
            "files": file_list[:50],  # Limit to first 50 files
            "total_files": len(file_list),
            "size_mb": sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
        }

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[W&B Local] {self.address_string()} - {format % args}")


def main():
    """Start the W&B local utilities server."""
    server = HTTPServer((WandBLocalConfig.HOST, WandBLocalConfig.PORT), WandBLocalHandler)

    print("=" * 60)
    print("W&B Local Utilities Server (Phase 1)")
    print("=" * 60)
    print(f"Host: {WandBLocalConfig.HOST}")
    print(f"Port: {WandBLocalConfig.PORT}")
    print(f"Runs Directory: {WandBLocalConfig.RUNS_DIR}")
    print("-" * 60)
    print("Mode: OFFLINE (Phase 1)")
    print("To sync runs when online: wandb sync ./outputs/wandb/offline-*")
    print("-" * 60)
    print("PHASE 2/3 TODO: Deploy wandb/local for self-hosted tracking")
    print("=" * 60)
    print(f"\nServer running at http://{WandBLocalConfig.HOST}:{WandBLocalConfig.PORT}")
    print("Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
