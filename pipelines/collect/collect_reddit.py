#!/usr/bin/env python3
"""
Data Collector - Reddit & Gallery-DL Integration
=================================================
Downloads images and videos from Reddit subreddits and other sources
using gallery-dl for building ML training datasets.

Usage:
    python collect.py --subreddit earthporn --limit 100
    python collect.py --subreddit "earthporn,cityporn" --limit 50 --sort top --time week
    python collect.py --url "https://reddit.com/r/art/top"
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Optional imports - use no-op decorators if not available
try:
    from utils.decorators import flow, task
except ImportError:
    # No-op decorators if dependencies missing
    def flow(func):
        return func
    def task(func):
        return func

try:
    from utils.hydra_aim import init_aim_from_hydra
except ImportError:
    def init_aim_from_hydra(*args, **kwargs):
        return None

try:
    from utils.manifest import record_collection_manifest
except ImportError:
    def record_collection_manifest(*args, **kwargs):
        pass

# Optional Hydra integration
try:
    import hydra
    from omegaconf import DictConfig
except ImportError:
    hydra = None
    DictConfig = Any


@task(name="collect-subreddit", retries=2, retry_delay_seconds=30)
def collect_subreddit_task(collector, subreddit: str, limit: int, sort: str, time_range: str, media_filter: Optional[str] = None):
    """Prefect task wrapper for subreddit collection."""
    return collector.collect_subreddit(subreddit, limit, sort, time_range, media_filter)


class DataCollector:
    """Collects images/videos from various sources using gallery-dl."""

    def __init__(
        self,
        output_dir: Path,
        config_path: Optional[Path] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = config_path or Path(__file__).parent / "gallery-dl.conf"
        self.metadata_file = self.output_dir / "collection_metadata.json"

        # Track collected items
        self.collected_items = []

    def collect_subreddit(
        self,
        subreddit: str,
        limit: int = 100,
        sort: str = "hot",
        time_range: str = "all",
        media_filter: Optional[str] = None
    ) -> list:
        """
        Collect images from a Reddit subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            limit: Maximum number of posts to fetch
            sort: Sort method (hot, new, top, rising)
            time_range: Time range for top/controversial (hour, day, week, month, year, all)
            media_types: Filter by media type (image, video, etc.)
        """
        print(f"\n{'='*60}")
        print(f"Collecting from r/{subreddit}")
        print(f"{'='*60}")
        print(f"Limit: {limit} | Sort: {sort} | Time: {time_range}")

        # Build URL
        url = f"https://reddit.com/r/{subreddit}/{sort}"
        if sort in ["top", "controversial"]:
            url += f"?t={time_range}"

        # Output directory for this subreddit
        sub_output = self.output_dir / subreddit
        sub_output.mkdir(parents=True, exist_ok=True)

        # Build gallery-dl command
        cmd = [
            "gallery-dl",
            "--dest", str(sub_output),
            "--range", f"1-{limit}",
            "--write-metadata",
            "--write-info-json",
            "-o", "filename={id}_{title:.50}.{extension}",
            "-o", f"directory=[]",  # Flat structure
        ]

        # Add config if exists
        if self.config_path.exists():
            cmd.extend(["--config", str(self.config_path)])

        # Filter media types if specified
        if media_filter:
            cmd.extend(["--filter", media_filter])

        cmd.append(url)

        print(f"\nRunning: {' '.join(cmd)}")
        print("-" * 60)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Warning: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("Error: Collection timed out")
        except Exception as e:
            print(f"Error: {e}")

        # Count collected files
        collected = list(sub_output.glob("*.*"))
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        video_extensions = {".mp4", ".webm", ".mov"}

        images = [f for f in collected if f.suffix.lower() in image_extensions]
        videos = [f for f in collected if f.suffix.lower() in video_extensions]

        print(f"\n{'='*60}")
        print(f"Collection complete: {len(images)} images, {len(videos)} videos")
        print(f"Output: {sub_output}")
        print(f"{'='*60}")

        # Record metadata
        collection_info = {
            "subreddit": subreddit,
            "url": url,
            "limit": limit,
            "sort": sort,
            "time_range": time_range,
            "collected_at": datetime.now().isoformat(),
            "images": len(images),
            "videos": len(videos),
            "output_dir": str(sub_output)
        }
        self.collected_items.append(collection_info)

        return images + videos

    def collect_url(self, url: str) -> list:
        """Collect from any gallery-dl supported URL."""
        print(f"\n{'='*60}")
        print(f"Collecting from URL: {url}")
        print(f"{'='*60}")

        # Determine output subdirectory from URL
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        sub_output = self.output_dir / f"url_{url_hash}"
        sub_output.mkdir(parents=True, exist_ok=True)

        cmd = [
            "gallery-dl",
            "--dest", str(sub_output),
            "--write-metadata",
            "--write-info-json",
            url
        ]

        if self.config_path.exists():
            cmd.extend(["--config", str(self.config_path)])

        print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            print(result.stdout if result.returncode == 0 else result.stderr)
        except Exception as e:
            print(f"Error: {e}")

        collected = list(sub_output.glob("*.*"))
        print(f"Collected {len(collected)} files to {sub_output}")

        return collected

    def save_metadata(self):
        """Save collection metadata to JSON."""
        metadata = {
            "collections": self.collected_items,
            "total_collections": len(self.collected_items),
            "generated_at": datetime.now().isoformat()
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata saved to: {self.metadata_file}")

    def list_supported_sites(self):
        """List all sites supported by gallery-dl."""
        result = subprocess.run(
            ["gallery-dl", "--list-extractors"],
            capture_output=True,
            text=True
        )
        print("Supported sites (partial list):")
        print("-" * 40)
        # Show first 50 lines
        lines = result.stdout.strip().split('\n')[:50]
        for line in lines:
            print(f"  {line}")
        print(f"  ... and {len(result.stdout.strip().split(chr(10))) - 50} more")


def main():
    parser = argparse.ArgumentParser(
        description="Collect images/videos from Reddit and other sources"
    )

    parser.add_argument(
        "--subreddit", "-s",
        type=str,
        help="Subreddit name(s) to collect from (comma-separated)"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        help="Direct URL to collect from"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Maximum items to collect per source (default: 100)"
    )
    parser.add_argument(
        "--sort",
        choices=["hot", "new", "top", "rising"],
        default="hot",
        help="Sort method for Reddit (default: hot)"
    )
    parser.add_argument(
        "--time",
        choices=["hour", "day", "week", "month", "year", "all"],
        default="all",
        help="Time range for top/controversial (default: all)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./data/collected"),
        help="Output directory (default: ./data/collected)"
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only collect images (no videos)"
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="List supported sites and exit"
    )
    parser.add_argument(
        "--hydra",
        action="store_true",
        help="Use Hydra config (conf/pipeline/collect.yaml) instead of CLI flags"
    )

    args = parser.parse_args()

    collector = DataCollector(output_dir=args.output_dir)

    if args.list_sites:
        collector.list_supported_sites()
        return

    if not args.subreddit and not args.url and not args.hydra:
        parser.print_help()
        print("\nExample usage:")
        print("  python collect.py --subreddit earthporn --limit 50")
        print("  python collect.py --subreddit 'art,pics' --sort top --time week")
        print("  python collect.py --url 'https://reddit.com/r/art/top'")
        print("  python collect.py --hydra  # uses conf/pipeline/collect.yaml")
        return

    media_filter = None
    if args.images_only:
        exts = ["jpg", "jpeg", "png", "gif", "webp"]
        media_filter = f"extension in ({', '.join([repr(e) for e in exts])})"

    if args.subreddit:
        # Handle multiple subreddits
        subreddits = [s.strip() for s in args.subreddit.split(",")]
        for sub in subreddits:
            collector.collect_subreddit(
                subreddit=sub,
                limit=args.limit,
                sort=args.sort,
                time_range=args.time,
                media_filter=media_filter
            )

    if args.url:
        collector.collect_url(args.url)

    collector.save_metadata()
    # Write collection manifest
    total_images = sum(item.get("images", 0) for item in collector.collected_items)
    total_videos = sum(item.get("videos", 0) for item in collector.collected_items)
    record_collection_manifest(
        name="reddit-collection",
        output_dir=args.output_dir,
        source={
            "type": "reddit" if args.subreddit else "url",
            "subreddit": args.subreddit,
            "url": args.url,
            "limit": args.limit,
            "sort": args.sort,
            "time_filter": args.time,
        },
        counts={"images": total_images, "videos": total_videos},
        cfg=None,
    )

    print("\n" + "=" * 60)
    print("Collection Complete!")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Caption images: python -m pipelines.annotate.caption")
    print("  2. View in FiftyOne: python -m pipelines.annotate.create_dataset")


@flow(name="data-collection-pipeline", log_prints=True)
def run_collection_flow(
    subreddits: list[str],
    output_dir: Path,
    limit: int = 100,
    sort: str = "hot",
    time_range: str = "all",
    images_only: bool = False
) -> list:
    """
    Prefect flow for data collection.

    (Prefect removed; run directly with python -m pipelines.collect.collect_reddit)
    """
    collector = DataCollector(output_dir=output_dir)
    media_filter = None
    if images_only:
        media_filter = "extension in ('jpg','jpeg','png','gif','webp')"

    all_files = []
    for sub in subreddits:
        files = collect_subreddit_task(collector, sub, limit, sort, time_range, media_filter)
        all_files.extend(files)

    collector.save_metadata()
    return all_files


if __name__ == "__main__":
    # Support Hydra-driven runs with --hydra flag
    if hydra is not None and "--hydra" in sys.argv:
        # Remove the flag so Hydra doesn't treat it as an unknown argument
        sys.argv = [arg for arg in sys.argv if arg != "--hydra"]

        @hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
        def hydra_main(cfg: "DictConfig"):  # type: ignore[misc]
            collect_cfg = cfg.get("pipeline", {}).get("collect", {})
            source = collect_cfg.get("source", {})
            filter_cfg = collect_cfg.get("filter", {})
            output_cfg = collect_cfg.get("output", {})

            subreddit = source.get("subreddit")
            url = source.get("url")
            limit = source.get("limit", 100)
            sort = source.get("sort", "hot")
            time_filter = source.get("time_filter", "all")
            allowed_ext = filter_cfg.get("allowed_extensions", None)
            # gallery-dl expects a string expression; pass None to disable
            media_filter = None
            if allowed_ext:
                # Convert [".jpg", ".png"] -> "extension in ['jpg','png']"
                exts = [ext.lstrip(".") for ext in allowed_ext]
                media_filter = f"extension in ({', '.join([repr(e) for e in exts])})"

            output_dir = Path(output_cfg.get("dir", "./data/collected"))
            collector = DataCollector(output_dir=output_dir)

            if subreddit:
                subs = [s.strip() for s in str(subreddit).split(",")]
                for sub in subs:
                    collect_subreddit_task(
                        collector=collector,
                        subreddit=sub,
                        limit=limit,
                        sort=sort,
                        time_range=time_filter,
                        media_filter=media_filter,
                    )

            if url:
                collector.collect_url(url)

            # Log config + git state into AIM
            init_aim_from_hydra(cfg, run_name=f"collect-{subreddit or 'url'}", experiment="collect")
            collector.save_metadata()

            total_images = sum(item.get("images", 0) for item in collector.collected_items)
            total_videos = sum(item.get("videos", 0) for item in collector.collected_items)
            record_collection_manifest(
                name="reddit-collection",
                output_dir=output_dir,
                source={
                    "type": "reddit" if subreddit else "url",
                    "subreddit": subreddit,
                    "url": url,
                    "limit": limit,
                    "sort": sort,
                    "time_filter": time_filter,
                },
                counts={"images": total_images, "videos": total_videos},
                cfg=cfg,
            )

        hydra_main()
    else:
        main()
