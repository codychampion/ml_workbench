#!/usr/bin/env python3
"""
Quick image scraper for testing - downloads a few images from Reddit
"""
import praw
import requests
import os
from pathlib import Path
from urllib.parse import urlparse
import time

def download_image(url: str, output_dir: Path) -> bool:
    """Download a single image"""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()

        # Get filename from URL
        filename = Path(urlparse(url).path).name
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            filename += '.jpg'

        output_path = output_dir / filename
        with open(output_path, 'wb') as f:
            f.write(response.content)

        return True
    except Exception as e:
        print(f"  ⚠️  Failed to download {url}: {e}")
        return False

def scrape_reddit(subreddit: str, limit: int, output_dir: str):
    """Scrape images from a subreddit"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📥 Scraping r/{subreddit} for {limit} images...")
    print(f"   Output: {output_path.absolute()}")

    # Try to use PRAW if credentials available
    try:
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID', 'dummy'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET', 'dummy'),
            user_agent='lora_trainer/1.0'
        )

        downloaded = 0
        for submission in reddit.subreddit(subreddit).hot(limit=limit * 3):
            if downloaded >= limit:
                break

            url = submission.url
            if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                print(f"  Downloading: {submission.title[:50]}...")
                if download_image(url, output_path):
                    downloaded += 1
                    time.sleep(0.5)  # Be nice to Reddit

        print(f"\n✅ Downloaded {downloaded} images to {output_path}")
        return downloaded

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 To use Reddit scraping, set these environment variables:")
        print("   REDDIT_CLIENT_ID=your_client_id")
        print("   REDDIT_CLIENT_SECRET=your_secret")
        print("\n   Or use the full collect_reddit.py pipeline")
        return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick Reddit image scraper")
    parser.add_argument("--subreddit", default="fallout", help="Subreddit to scrape")
    parser.add_argument("--limit", type=int, default=20, help="Number of images")
    parser.add_argument("--output", default="./data/scraped/test", help="Output directory")

    args = parser.parse_args()

    count = scrape_reddit(args.subreddit, args.limit, args.output)

    if count > 0:
        print(f"\n🎯 Ready to train! Run:")
        print(f"   python scripts/train_lora_auto.py --source {args.output} --concept {args.subreddit}")
