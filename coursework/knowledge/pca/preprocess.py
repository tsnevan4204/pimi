"""
preprocess.py

Turns a folder of PNG images into one flat feature vector per image, all
the same length, ready to feed into a hand-written PCA implementation
(e.g. in C++).

WHAT IT DOES, PER IMAGE:
1. Load the PNG.
2. Convert to grayscale (1 value/pixel) or RGB (3 values/pixel).
3. Resize to a fixed size×size square so every vector has the same length.
4. Flatten row-major into a 1-D vector.
5. Normalize pixel values from 0..255 to 0.0..1.0 floats.

WHY GRAYSCALE + DOWNSCALE (default 64x64):
Hand-coded PCA usually builds a D x D covariance matrix, where D is the
number of features per image. RGB at 128x128 is D = 49,152 -> a ~19 GB
covariance matrix. Grayscale at 64x64 is D = 4,096 -> a ~134 MB matrix,
which is very manageable. You have far more features than images (~96),
so downscaling loses little and keeps PCA tractable. Pass --rgb and/or
--size to change this.

OUTPUT (default: JSON):
{
  "channels": 1,
  "size": 64,
  "width": 64,
  "height": 64,
  "n_samples": 96,
  "n_features": 4096,          # = size*size*channels
  "layout": "row-major, then channel-interleaved if rgb (r,g,b,r,g,b,...)",
  "normalized": "pixel/255.0 in [0,1]",
  "centered": false,
  "ids": ["1232632452458901", ...],   # one id per row, from the filename
  "vectors": [[...], [...], ...]        # n_samples rows, each n_features long
}

ALTERNATIVE OUTPUT (--format csv):
A headerless CSV, one image per line, n_features comma-separated floats.
Trivial to read in C++ with no JSON library. A sidecar file
<output>.meta.json holds the shape info (n_samples, n_features, ids, ...).

USAGE:
    python preprocess.py --indir ./son_gifs/pngs --out ./pca_input.json
    python preprocess.py --indir ./son_gifs/pngs --out ./pca_input.csv --format csv
    python preprocess.py --indir ./son_gifs/pngs --rgb --size 32 --out ./pca_rgb.json
    python preprocess.py --indir ./son_gifs/pngs --center --out ./pca_input.json
"""

import argparse
import glob
import json
import os
import sys

from PIL import Image


def image_to_vector(path, size, rgb):
    """Load one image and return a flat list of normalized floats."""
    im = Image.open(path)
    im = im.convert("RGB" if rgb else "L")
    im = im.resize((size, size))

    pixels = list(im.getdata())  # list of (r,g,b) tuples if rgb, else ints
    if rgb:
        # flatten (r,g,b) tuples into r,g,b,r,g,b,...
        flat = [c / 255.0 for px in pixels for c in px]
    else:
        flat = [px / 255.0 for px in pixels]
    return flat


def main():
    parser = argparse.ArgumentParser(
        description="Flatten a folder of PNGs into equal-length vectors for PCA."
    )
    parser.add_argument("--indir", default="./son_gifs/pngs", help="Folder of PNG images")
    parser.add_argument("--out", default="./pca_input.json", help="Output file path")
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Resize every image to size x size (default 64). Controls vector length.",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Keep 3 channels per pixel (RGB). Default is grayscale (1 channel).",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Subtract the per-feature mean across all images (mean-centering). "
        "Off by default so you can do centering inside your own PCA code.",
    )
    parser.add_argument(
        "--format",
        default="json",
        choices=["json", "csv"],
        help="Output format. json = one file with everything. "
        "csv = headerless matrix + a <out>.meta.json sidecar (easier for C++).",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.indir, "*.png")))
    if not paths:
        sys.exit(f"No PNGs found in {args.indir}")

    channels = 3 if args.rgb else 1
    n_features = args.size * args.size * channels

    print(
        f"Processing {len(paths)} images -> {args.size}x{args.size} "
        f"{'RGB' if args.rgb else 'grayscale'} "
        f"({n_features} features each)..."
    )

    ids = []
    vectors = []
    for i, path in enumerate(paths):
        try:
            vec = image_to_vector(path, args.size, args.rgb)
        except Exception as e:
            print(f"  skipping {path}: {e}")
            continue
        if len(vec) != n_features:
            print(f"  skipping {path}: got {len(vec)} features, expected {n_features}")
            continue
        ids.append(os.path.splitext(os.path.basename(path))[0])
        vectors.append(vec)
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(paths)}")

    if not vectors:
        sys.exit("No vectors were produced — check the input images.")

    if args.center:
        n = len(vectors)
        means = [0.0] * n_features
        for vec in vectors:
            for j, v in enumerate(vec):
                means[j] += v
        means = [m / n for m in means]
        for vec in vectors:
            for j in range(n_features):
                vec[j] -= means[j]
        print("Applied per-feature mean-centering.")

    meta = {
        "channels": channels,
        "size": args.size,
        "width": args.size,
        "height": args.size,
        "n_samples": len(vectors),
        "n_features": n_features,
        "layout": (
            "row-major; channel-interleaved (r,g,b,r,g,b,...)"
            if args.rgb
            else "row-major grayscale"
        ),
        "normalized": "pixel/255.0 in [0,1]",
        "centered": bool(args.center),
        "ids": ids,
    }

    if args.format == "json":
        out = dict(meta)
        out["vectors"] = vectors
        with open(args.out, "w") as f:
            json.dump(out, f)
        print(f"\nWrote {len(vectors)} vectors of length {n_features} to {args.out}")
    else:  # csv
        with open(args.out, "w") as f:
            for vec in vectors:
                f.write(",".join(f"{v:.6f}" for v in vec))
                f.write("\n")
        meta_path = args.out + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nWrote {len(vectors)}x{n_features} matrix to {args.out}")
        print(f"Wrote shape/id metadata to {meta_path}")

    print(f"n_samples = {len(vectors)}, n_features = {n_features}")


if __name__ == "__main__":
    main()
