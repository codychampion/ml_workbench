"""
Git Utilities - Shared Git Information Extraction
==================================================
Provides git repository information for experiment tracking and provenance.

Usage:
    from utils.git_utils import get_git_info

    git_info = get_git_info()
    print(f"Commit: {git_info['commit_short']} on {git_info['branch']}")
"""

import subprocess
from typing import Dict, Any, List


def _run_git(args: List[str], timeout: int = 10) -> str:
    """
    Run a git command and return output.

    Args:
        args: Git command arguments (without 'git' prefix)
        timeout: Command timeout in seconds

    Returns:
        Command output (stdout) or empty string on error
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def get_git_info() -> Dict[str, Any]:
    """
    Get current git state for experiment traceability.

    Returns dict with commit hash, branch, author, dirty status, file changes,
    and repo URLs for linking to commits.
    """
    commit = _run_git(["rev-parse", "HEAD"])
    commit_short = _run_git(["rev-parse", "--short", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit_message = _run_git(["log", "-1", "--format=%s"])
    commit_author = _run_git(["log", "-1", "--format=%an <%ae>"])
    commit_date = _run_git(["log", "-1", "--format=%ci"])

    # Check if working directory has uncommitted changes
    dirty = bool(_run_git(["status", "--porcelain"]))

    # Get list of files changed in the current commit
    modified_files = _run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"])
    modified_list = modified_files.split("\n") if modified_files else []

    # Get uncommitted changes if dirty
    uncommitted = []
    if dirty:
        uncommitted_output = _run_git(["status", "--porcelain"])
        uncommitted = [line[3:] for line in uncommitted_output.split("\n") if line]

    # Try to get remote URL for building commit links
    remote_url = _run_git(["remote", "get-url", "origin"])
    repo_url = ""
    if remote_url:
        # Convert git@github.com:user/repo.git to https://github.com/user/repo
        if remote_url.startswith("git@"):
            repo_url = remote_url.replace("git@", "https://").replace(":", "/").rstrip(".git")
        elif remote_url.startswith("https://"):
            repo_url = remote_url.rstrip(".git")

    return {
        "commit": commit,
        "commit_short": commit_short,
        "branch": branch,
        "message": commit_message,
        "commit_message": commit_message,  # Alias for compatibility
        "author": commit_author,
        "commit_author": commit_author,  # Alias for compatibility
        "commit_date": commit_date,
        "dirty": dirty,
        "uncommitted_files": uncommitted,
        "modified_files": modified_list,
        "repo_url": repo_url,
        "commit_url": f"{repo_url}/commit/{commit}" if repo_url and commit else "",
        "diff_from_main": f"{repo_url}/compare/main...{commit}" if repo_url and commit else "",
    }
