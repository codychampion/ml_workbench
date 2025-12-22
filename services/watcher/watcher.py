#!/usr/bin/env python3
"""
Watcher service to sync AIM runs and service metadata into the knowledge base.
"""

import os
import time
from pathlib import Path

from utils.knowledge_sync import sync_all

DEFAULT_INTERVAL = int(os.environ.get("WATCH_INTERVAL_SECONDS", "300"))
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "./knowledge"))


def main():
    interval = DEFAULT_INTERVAL
    print(f"[Watcher] Starting knowledge sync loop (interval={interval}s)")
    print(f"[Watcher] Knowledge dir: {KNOWLEDGE_DIR}")

    while True:
        try:
            sync_all()
        except Exception as e:
            print(f"[Watcher] Error during sync: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
