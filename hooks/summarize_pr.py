#!/usr/bin/env python3
"""
Git PR/Commit Summarizer with LLM
==================================
Uses LiteLLM to generate intelligent summaries of git changes
for ingestion into the knowledge base.

Usage:
    python -m hooks.summarize_pr                    # Summarize latest merge
    python -m hooks.summarize_pr HEAD~5..HEAD       # Summarize commit range
    python -m hooks.summarize_pr --pr 123           # Summarize GitHub PR
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configuration
LITELLM_BASE = os.environ.get("LITELLM_API_BASE", "http://localhost:4000")
LITELLM_KEY = os.environ.get("LITELLM_API_KEY", "sk-mlops-dev-key")
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "./knowledge/git-summaries"))
MODEL = os.environ.get("SUMMARIZE_MODEL", "gpt-3.5-turbo")


def run_git_command(cmd: list) -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            ["git"] + cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def get_commit_range_info(commit_range: str) -> dict:
    """Get information about a commit range."""
    # Parse commit range (e.g., "HEAD~5..HEAD" or "abc123")
    if ".." in commit_range:
        start, end = commit_range.split("..")
    else:
        start = f"{commit_range}^"
        end = commit_range

    info = {
        "range": commit_range,
        "commits": run_git_command(["log", "--oneline", f"{start}..{end}"]),
        "files_changed": run_git_command(["diff", "--name-only", start, end]),
        "stats": run_git_command(["diff", "--stat", start, end]),
        "diff_summary": run_git_command(["diff", "--shortstat", start, end]),
    }

    # Get detailed diff (truncated for LLM context)
    full_diff = run_git_command(["diff", start, end])
    info["diff"] = full_diff[:8000] + "..." if len(full_diff) > 8000 else full_diff

    return info


def get_github_pr_info(pr_number: int) -> dict:
    """Get PR information from GitHub using gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "pr", "view", str(pr_number), "--json",
             "title,body,author,commits,files,additions,deletions"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"Error fetching PR: {e}")
    return None


def generate_ai_summary(info: dict) -> str:
    """Generate AI summary using LiteLLM."""
    if not REQUESTS_AVAILABLE:
        return "*Install requests package for AI summaries: pip install requests*"

    prompt = f"""Analyze this git diff and provide a concise technical summary.

COMMITS:
{info.get('commits', 'N/A')}

FILES CHANGED:
{info.get('files_changed', 'N/A')}

STATS:
{info.get('stats', 'N/A')}

DIFF (truncated):
{info.get('diff', 'N/A')[:4000]}

Provide:
1. **Summary** (2-3 sentences describing the overall change)
2. **Key Changes** (bullet points of main modifications)
3. **Impact** (what parts of the system are affected)
4. **Review Notes** (anything to watch out for)

Be concise and technical."""

    try:
        response = requests.post(
            f"{LITELLM_BASE}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {LITELLM_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a senior software engineer reviewing code changes. Be concise and technical."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.3
            },
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"*AI summary unavailable: {response.status_code}*"

    except Exception as e:
        return f"*AI summary error: {e}*"


def create_summary_file(info: dict, ai_summary: str, output_path: Path = None) -> Path:
    """Create markdown summary file."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now()
    filename = f"{timestamp.strftime('%Y%m%d-%H%M%S')}-summary.md"
    filepath = output_path or (KNOWLEDGE_DIR / filename)

    content = f"""---
type: git-summary
date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
range: {info.get('range', 'unknown')}
tags: [git, code-review, changelog]
---

# Code Change Summary

**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Range:** `{info.get('range', 'unknown')}`

## Statistics

```
{info.get('diff_summary', 'N/A')}
```

## Commits

```
{info.get('commits', 'N/A')}
```

## AI Analysis

{ai_summary}

## Files Changed

```
{info.get('files_changed', 'N/A')}
```

---
*Auto-generated by summarize_pr.py*
"""

    filepath.write_text(content)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate AI summaries of git changes")
    parser.add_argument("range", nargs="?", default="HEAD^..HEAD",
                        help="Commit range (e.g., HEAD~5..HEAD)")
    parser.add_argument("--pr", type=int, help="GitHub PR number")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI summary")

    args = parser.parse_args()

    if args.pr:
        print(f"Fetching PR #{args.pr}...")
        pr_info = get_github_pr_info(args.pr)
        if pr_info:
            info = {
                "range": f"PR #{args.pr}",
                "commits": "\n".join(c.get("messageHeadline", "") for c in pr_info.get("commits", [])),
                "files_changed": "\n".join(f.get("path", "") for f in pr_info.get("files", [])),
                "stats": f"+{pr_info.get('additions', 0)} -{pr_info.get('deletions', 0)}",
                "diff_summary": f"{len(pr_info.get('files', []))} files changed",
                "diff": pr_info.get("body", "")
            }
        else:
            print("Could not fetch PR info")
            sys.exit(1)
    else:
        print(f"Analyzing {args.range}...")
        info = get_commit_range_info(args.range)

    # Generate AI summary
    if args.no_ai:
        ai_summary = "*AI summary skipped*"
    else:
        print("Generating AI summary...")
        ai_summary = generate_ai_summary(info)

    # Create file
    output_path = Path(args.output) if args.output else None
    filepath = create_summary_file(info, ai_summary, output_path)

    print(f"\nSummary created: {filepath}")
    print("\n" + "=" * 60)
    print(ai_summary)


if __name__ == "__main__":
    main()
