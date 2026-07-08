"""
klipy_gif_scraper.py

Downloads GIF search results using the Klipy API (Tenor's API was fully
shut down by Google on June 30, 2026 — Klipy is the closest free
drop-in replacement, and even ships a Tenor-migration mode), converts
each GIF to a representative PNG frame, and saves everything to a
local directory — set up for a downstream PCA analysis.

WHY AN API INSTEAD OF SCRAPING THE TENOR SEARCH PAGE:
tenor.com/search/<query> is a JavaScript-rendered React page — GIFs
aren't in the raw HTML, they're fetched via XHR after the page loads.
A requests+BeautifulSoup scraper of that page would return basically
nothing. Using a proper GIF API gives clean structured data (direct
.gif URLs, dimensions, IDs) instead.

SETUP:
1. Get a free Klipy API key: https://klipy.com/developers
   (a "Test Key" is issued instantly, good for 100 calls/hour — plenty
   for this script. You can request a production key later if needed.)
2. Set it as an environment variable before running:
       export KLIPY_API_KEY="your_key_here"

IMPORTANT — RUN THIS FIRST:
Klipy's own docs site is JS-rendered too, so I could not scrape the
exact field names of their search response ahead of time. Run this
script once with --debug first:

    python klipy_gif_scraper.py --query son --limit 5 --debug

That will print the raw JSON of one API response. If the download step
fails or produces 0 results, check that printed JSON against the
`extract_gif_url()` function below and adjust the field names to match
(they're clearly marked with a comment). This is a 30-second fix if
Klipy's schema differs slightly from what's assumed here.

USAGE:
    python klipy_gif_scraper.py --query son --limit 250 --outdir ./son_gifs

OUTPUT:
    <outdir>/gifs/*.gif          - raw downloaded gifs
    <outdir>/pngs/*.png          - one representative PNG frame per gif
    <outdir>/metadata.csv        - id, query, width, height, frame_count, url
"""

import argparse
import csv
import json
import os
import sys
import time
import uuid
from io import BytesIO

import requests
from PIL import Image, ImageSequence

KLIPY_BASE_URL = "https://api.klipy.com/api/v1"


def get_api_key():
    key = os.environ.get("KLIPY_API_KEY")
    if not key:
        sys.exit(
            "ERROR: No Klipy API key found.\n"
            "Set it with: export KLIPY_API_KEY='your_key_here'\n"
            "Get one free at: https://klipy.com/developers"
        )
    return key


def extract_gif_url(item):
    """
    Pull a direct .gif URL and dimensions out of a single Klipy search
    result item.

    Verified against a live Klipy response (--debug). The real schema is:

        item["file"][quality]["gif"] = {"url", "width", "height", "size"}

    where `quality` is one of "hd", "md", "sm", "xs" (each quality also
    contains "webp", "jpg", "mp4", "webm" variants). We prefer md, then
    fall back through the other qualities. A few defensive fallbacks for
    older/other shapes are kept below.
    """
    # Verified shape: item["file"][quality]["gif"]["url"]
    for media_key in ("file", "media", "formats", "media_formats"):
        media = item.get(media_key)
        if not isinstance(media, dict):
            continue
        for quality in ("md", "hd", "sm", "xs", "original"):
            quality_obj = media.get(quality)
            if isinstance(quality_obj, dict):
                gif = quality_obj.get("gif")
                if isinstance(gif, dict) and gif.get("url"):
                    return gif["url"], gif.get("width"), gif.get("height")
        # Fallback: a quality bucket that is itself the gif format
        for quality in ("gif", "original", "hd", "md", "sm"):
            fmt = media.get(quality)
            if isinstance(fmt, dict) and fmt.get("url"):
                return fmt["url"], fmt.get("width"), fmt.get("height")

    # Fallback: a flat url field directly on the item
    if item.get("url", "").endswith(".gif"):
        return item["url"], item.get("width"), item.get("height")

    # Fallback: "images" style list
    images = item.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"], first.get("width"), first.get("height")

    return None, None, None


def fetch_gif_results(query, limit, api_key, customer_id, debug=False):
    """
    Paginate through Klipy search results until `limit` results are
    collected or the API runs out. Returns a list of dicts with
    id, url, width, height.
    """
    results = []
    page = 1
    per_page = 24  # Klipy's default page size in most of their endpoints

    while len(results) < limit:
        url = f"{KLIPY_BASE_URL}/{api_key}/gifs/search"
        params = {
            "q": query,
            "customer_id": customer_id,
            "page": page,
            "per_page": min(per_page, limit - len(results)),
        }

        resp = requests.get(url, params=params, timeout=30)
        if debug:
            print(f"\n[DEBUG] GET {resp.url}")
            print(f"[DEBUG] status: {resp.status_code}")
            print("[DEBUG] raw response (first 3000 chars):")
            print(resp.text[:3000])

        if resp.status_code != 200:
            print(f"  API error {resp.status_code}: {resp.text[:300]}")
            break

        data = resp.json()
        # results are commonly under data["data"]["data"] or data["data"] or data["results"]
        items = None
        for path in (("data", "data"), ("data",), ("results",)):
            node = data
            ok = True
            for key in path:
                if isinstance(node, dict) and key in node:
                    node = node[key]
                else:
                    ok = False
                    break
            if ok and isinstance(node, list):
                items = node
                break

        if not items:
            if debug:
                print("[DEBUG] Could not locate a results list in the response.")
                print("[DEBUG] Top-level keys were:", list(data.keys()))
            break

        for item in items:
            gif_url, w, h = extract_gif_url(item)
            if not gif_url:
                continue
            results.append(
                {
                    "id": item.get("id") or item.get("slug") or str(uuid.uuid4())[:8],
                    "url": gif_url,
                    "width": w,
                    "height": h,
                }
            )

        if debug:
            return results  # just return what we've got in debug mode

        if len(items) < params["per_page"]:
            break  # last page

        page += 1
        print(f"  fetched {len(results)}/{limit} result URLs so far...")
        time.sleep(0.2)

    return results[:limit]


def download_gif(url, timeout=30):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def gif_to_png(gif_bytes, frame_strategy="middle"):
    """
    Convert raw gif bytes to a single representative PNG (PIL Image, RGB).
    frame_strategy: "first", "middle", or "average".
    """
    im = Image.open(BytesIO(gif_bytes))
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(im)]

    if not frames:
        raise ValueError("No frames found in GIF")

    if frame_strategy == "first":
        return frames[0]
    elif frame_strategy == "middle":
        return frames[len(frames) // 2]
    elif frame_strategy == "average":
        import numpy as np

        arrs = [np.asarray(f, dtype="float32") for f in frames]
        avg = np.mean(arrs, axis=0).astype("uint8")
        return Image.fromarray(avg)
    else:
        raise ValueError(f"Unknown frame_strategy: {frame_strategy}")


def main():
    parser = argparse.ArgumentParser(description="Download GIFs via Klipy API and convert to PNG for PCA.")
    parser.add_argument("--query", default="son", help="Search term (default: 'son')")
    parser.add_argument("--limit", type=int, default=250, help="Max number of gifs to fetch")
    parser.add_argument("--outdir", default="./gifs_output", help="Output directory")
    parser.add_argument(
        "--frame-strategy",
        default="middle",
        choices=["first", "middle", "average"],
        help="Which frame(s) to use when converting gif -> png",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=128,
        help="Resize PNGs to NxN pixels (square) for consistent PCA input. 0 = no resize.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw API response for one page and exit (use this first!)",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    customer_id = str(uuid.uuid4())  # Klipy requires a stable per-user id; a random one is fine here

    if args.debug:
        print("Running in DEBUG mode: fetching one page and printing raw JSON...\n")
        results = fetch_gif_results(args.query, min(args.limit, 5), api_key, customer_id, debug=True)
        print(f"\n[DEBUG] Parsed {len(results)} usable gif URLs from that page.")
        if results:
            print("[DEBUG] Example parsed result:", json.dumps(results[0], indent=2))
        else:
            print(
                "[DEBUG] No gif URLs were extracted. Compare the raw response above to "
                "extract_gif_url() in this file and adjust the field names it looks for."
            )
        return

    gif_dir = os.path.join(args.outdir, "gifs")
    png_dir = os.path.join(args.outdir, "pngs")
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print(f"Searching for '{args.query}' (up to {args.limit} results)...")
    results = fetch_gif_results(args.query, args.limit, api_key, customer_id)
    print(f"Found {len(results)} gif results.")

    if not results:
        print("No results found. Try running with --debug to inspect the raw API response.")
        return

    metadata_path = os.path.join(args.outdir, "metadata.csv")
    with open(metadata_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "query", "width", "height", "frame_count", "url", "png_path"])

        for i, item in enumerate(results):
            gif_id = item["id"]
            url = item["url"]
            try:
                gif_bytes = download_gif(url)

                gif_path = os.path.join(gif_dir, f"{gif_id}.gif")
                with open(gif_path, "wb") as f:
                    f.write(gif_bytes)

                im = Image.open(BytesIO(gif_bytes))
                frame_count = getattr(im, "n_frames", 1)

                png_img = gif_to_png(gif_bytes, frame_strategy=args.frame_strategy)
                if args.resize > 0:
                    png_img = png_img.resize((args.resize, args.resize))

                png_path = os.path.join(png_dir, f"{gif_id}.png")
                png_img.save(png_path)

                writer.writerow(
                    [gif_id, args.query, item["width"], item["height"], frame_count, url, png_path]
                )

                if (i + 1) % 25 == 0:
                    print(f"  processed {i + 1}/{len(results)}")

            except Exception as e:
                print(f"  failed on {gif_id} ({url}): {e}")

            time.sleep(0.1)

    print(f"\nDone. GIFs in: {gif_dir}")
    print(f"PNGs in: {png_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
