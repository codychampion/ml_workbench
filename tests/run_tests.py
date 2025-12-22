#!/usr/bin/env python3
"""
MLOps Workbench Test Runner
============================
Comprehensive test runner with pretty output and service status reporting.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --quick            # Quick health checks only
    python tests/run_tests.py --category minio   # Run specific category
    python tests/run_tests.py --list             # List all test categories
    python tests/run_tests.py --status           # Show service status only
"""

import os
import sys
import argparse
import subprocess
import time
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Configuration
SERVICE_HOST = os.environ.get("SERVICE_HOST", "localhost")

# Service definitions for health checks
SERVICES = {
    # Core Infrastructure
    "minio": {"name": "MinIO", "url": f"http://{SERVICE_HOST}:9000/minio/health/live", "category": "core"},
    "mongodb": {"name": "MongoDB", "port": 27017, "category": "core"},
    "redis": {"name": "Redis", "port": 6379, "category": "core"},
    "postgres": {"name": "PostgreSQL", "port": 5432, "category": "core"},

    # Experiment Tracking
    "aim": {"name": "AIM", "url": f"http://{SERVICE_HOST}:43800/", "category": "mlops"},
    # Prefect removed from stack; keep placeholder disabled
    # "prefect": {"name": "Prefect", "url": f"http://{SERVICE_HOST}:4200/api/health", "category": "mlops"},
    "great_expectations": {"name": "Great Expectations", "url": f"http://{SERVICE_HOST}:8084/", "category": "mlops"},

    # Annotation
    "label_studio": {"name": "Label Studio", "url": f"http://{SERVICE_HOST}:8081/health", "category": "annotation"},
    "cvat": {"name": "CVAT", "url": f"http://{SERVICE_HOST}:8082/", "category": "annotation"},
    "spotlight": {"name": "Spotlight", "url": f"http://{SERVICE_HOST}:8083/", "category": "annotation"},
    "fiftyone": {"name": "FiftyOne", "url": f"http://{SERVICE_HOST}:5151/", "category": "annotation"},

    # Knowledge
    "khoj": {"name": "Khoj", "url": f"http://{SERVICE_HOST}:42110/api/health", "category": "knowledge"},
    "couchdb": {"name": "CouchDB", "url": f"http://{SERVICE_HOST}:5984/_up", "auth": ("obsidian", "mlops-dev-password"), "category": "knowledge"},
    "zotero": {"name": "Zotero", "url": f"http://{SERVICE_HOST}:8085/health", "category": "knowledge"},

    # Other
    "vault": {"name": "Vault", "url": f"http://{SERVICE_HOST}:8200/v1/sys/health", "category": "security"},
    "litellm": {"name": "LiteLLM", "url": f"http://{SERVICE_HOST}:4000/health", "category": "llm"},
    "comfyui": {"name": "ComfyUI", "url": f"http://{SERVICE_HOST}:8188/", "category": "generation"},
}

# Test categories
TEST_CATEGORIES = {
    "health": "tests/integration/test_services_health.py",
    "minio": "tests/integration/test_minio_integration.py",
    "knowledge": "tests/integration/test_knowledge_stack.py",
    "mlops": "tests/integration/test_mlops_services.py",
}


def check_service_health(service_key: str, service_config: dict) -> tuple:
    """Check if a service is healthy."""
    if not REQUESTS_AVAILABLE:
        return service_key, "unknown", "requests not installed"

    if "url" in service_config:
        try:
            kwargs = {"timeout": 5}
            if "auth" in service_config:
                kwargs["auth"] = service_config["auth"]

            response = requests.get(service_config["url"], **kwargs)
            if response.status_code in [200, 204]:
                return service_key, "healthy", f"OK ({response.status_code})"
            else:
                return service_key, "unhealthy", f"Status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return service_key, "down", "Connection refused"
        except requests.exceptions.Timeout:
            return service_key, "timeout", "Timeout"
        except Exception as e:
            return service_key, "error", str(e)[:50]
    elif "port" in service_config:
        # TCP port check
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((SERVICE_HOST, service_config["port"]))
            sock.close()
            if result == 0:
                return service_key, "healthy", "Port open"
            else:
                return service_key, "down", "Port closed"
        except Exception as e:
            return service_key, "error", str(e)[:50]

    return service_key, "unknown", "No check configured"


def check_all_services_parallel() -> Dict[str, tuple]:
    """Check all services in parallel."""
    results = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(check_service_health, key, config): key
            for key, config in SERVICES.items()
        }

        for future in as_completed(futures):
            key, status, message = future.result()
            results[key] = (status, message)

    return results


def print_service_status(results: Dict[str, tuple]):
    """Print service status table."""
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Service Status")
        table.add_column("Service", style="cyan")
        table.add_column("Category", style="dim")
        table.add_column("Status")
        table.add_column("Details", style="dim")

        status_colors = {
            "healthy": "[green]✓ Healthy[/green]",
            "unhealthy": "[yellow]⚠ Unhealthy[/yellow]",
            "down": "[red]✗ Down[/red]",
            "timeout": "[yellow]⏱ Timeout[/yellow]",
            "error": "[red]! Error[/red]",
            "unknown": "[dim]? Unknown[/dim]",
        }

        for key in sorted(SERVICES.keys(), key=lambda k: SERVICES[k].get("category", "")):
            config = SERVICES[key]
            status, message = results.get(key, ("unknown", "Not checked"))
            table.add_row(
                config["name"],
                config.get("category", ""),
                status_colors.get(status, status),
                message
            )

        console.print(table)

        # Summary
        healthy = sum(1 for s, _ in results.values() if s == "healthy")
        total = len(results)
        console.print(f"\n[bold]Summary:[/bold] {healthy}/{total} services healthy")

    else:
        print("\n" + "=" * 60)
        print("SERVICE STATUS")
        print("=" * 60)

        for key, config in sorted(SERVICES.items(), key=lambda x: x[1].get("category", "")):
            status, message = results.get(key, ("unknown", "Not checked"))
            status_icon = {"healthy": "✓", "unhealthy": "⚠", "down": "✗", "error": "!"}.get(status, "?")
            print(f"{status_icon} {config['name']:20} [{config.get('category', ''):10}] {message}")

        print("=" * 60)


def run_pytest(category: Optional[str] = None, verbose: bool = False, quick: bool = False) -> int:
    """Run pytest for specified category."""
    cmd = ["python", "-m", "pytest", "-v"]

    if quick:
        # Only run quick health checks
        cmd.extend([
            "tests/integration/test_services_health.py::TestAllServicesQuickCheck",
            "--timeout=30"
        ])
    elif category:
        if category in TEST_CATEGORIES:
            cmd.append(TEST_CATEGORIES[category])
        else:
            print(f"Unknown category: {category}")
            print(f"Available: {', '.join(TEST_CATEGORIES.keys())}")
            return 1
    else:
        cmd.append("tests/integration/")

    cmd.extend([
        "-m", "integration",
        "--tb=short",
        f"--timeout={300 if not quick else 30}",
    ])

    if verbose:
        cmd.append("-vv")

    if RICH_AVAILABLE:
        console = Console()
        console.print(f"\n[bold blue]Running:[/bold blue] {' '.join(cmd)}\n")
    else:
        print(f"\nRunning: {' '.join(cmd)}\n")

    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="MLOps Workbench Test Runner")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick health checks only")
    parser.add_argument("--category", "-c", help="Run specific test category")
    parser.add_argument("--list", "-l", action="store_true", help="List test categories")
    parser.add_argument("--status", "-s", action="store_true", help="Show service status only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--host", default="localhost", help="Service host")

    args = parser.parse_args()

    global SERVICE_HOST
    SERVICE_HOST = args.host
    os.environ["SERVICE_HOST"] = args.host

    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit(
            "[bold cyan]MLOps Workbench Test Suite[/bold cyan]",
            subtitle=f"Host: {SERVICE_HOST}"
        ))
    else:
        print("\n" + "=" * 60)
        print("MLOps Workbench Test Suite")
        print(f"Host: {SERVICE_HOST}")
        print("=" * 60)

    if args.list:
        print("\nTest Categories:")
        for name, path in TEST_CATEGORIES.items():
            print(f"  {name:15} {path}")
        print("\nUsage: python tests/run_tests.py --category <name>")
        return 0

    # Always check service status first
    print("\nChecking service status...")
    results = check_all_services_parallel()
    print_service_status(results)

    if args.status:
        return 0

    # Check if enough services are up to run tests
    healthy_count = sum(1 for s, _ in results.values() if s == "healthy")
    if healthy_count < 3:
        if RICH_AVAILABLE:
            console = Console()
            console.print("\n[yellow]Warning: Few services are healthy. Tests may fail.[/yellow]")
            console.print("Make sure to run: [cyan]docker-compose up -d[/cyan]\n")
        else:
            print("\nWarning: Few services are healthy. Tests may fail.")
            print("Make sure to run: docker-compose up -d\n")

    # Run tests
    return run_pytest(
        category=args.category,
        verbose=args.verbose,
        quick=args.quick
    )


if __name__ == "__main__":
    sys.exit(main())
