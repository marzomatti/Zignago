#!/usr/bin/env python3
"""Find and remove unusably black images from the dataset.

Default behavior is a safe dry run:

    python src/cleanup_black_images.py

To actually delete detected black images:

    python src/cleanup_black_images.py --action delete --yes

The detector is intentionally conservative. An image is considered black only
when its mean brightness is low, most pixels are near black, and the 95th
percentile is still dark. This avoids removing valid bottle images that have a
dark background but still contain visible glass edges or highlights.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class ImageStats:
    path: Path
    width: int
    height: int
    mean: float
    std: float
    p95: float
    p99: float
    black_ratio: float
    bright_ratio: float
    is_black: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and optionally delete black images in the data folder."
    )
    parser.add_argument("--data-dir", default="data", help="Dataset root to scan.")
    parser.add_argument(
        "--output-dir",
        default="output/black_image_cleanup",
        help="Where to write reports and contact sheets.",
    )
    parser.add_argument(
        "--action",
        choices=["dry-run", "delete", "quarantine"],
        default="dry-run",
        help="dry-run writes reports only; delete removes files; quarantine moves files.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required for --action delete or --action quarantine.",
    )
    parser.add_argument(
        "--quarantine-dir",
        default="output/quarantined_black_images",
        help="Destination used when --action quarantine is selected.",
    )
    parser.add_argument(
        "--mean-threshold",
        type=float,
        default=30.0,
        help="Maximum grayscale mean for a black image.",
    )
    parser.add_argument(
        "--p95-threshold",
        type=float,
        default=55.0,
        help="Maximum 95th percentile grayscale value for a black image.",
    )
    parser.add_argument(
        "--black-pixel-threshold",
        type=int,
        default=25,
        help="Pixels below this grayscale value count as black.",
    )
    parser.add_argument(
        "--black-ratio-threshold",
        type=float,
        default=0.85,
        help="Minimum fraction of near-black pixels for a black image.",
    )
    parser.add_argument(
        "--bright-pixel-threshold",
        type=int,
        default=80,
        help="Pixels above this grayscale value count as bright.",
    )
    parser.add_argument(
        "--max-bright-ratio",
        type=float,
        default=0.04,
        help="Maximum fraction of bright pixels for a black image.",
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=256,
        help="Downscaled size used for statistics. Keeps scanning fast.",
    )
    parser.add_argument(
        "--contact-sheet-limit",
        type=int,
        default=80,
        help="Maximum detected black images shown in the contact sheet.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Optional subfolder to include. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.action in {"delete", "quarantine"} and not args.yes:
        raise SystemExit(
            f"Refusing to {args.action} without --yes. Run dry-run first, then "
            f"rerun with --action {args.action} --yes."
        )

    paths = list(iter_image_paths(data_dir, args.include))
    if not paths:
        raise SystemExit(f"No image files found under {data_dir}")

    print(f"Scanning {len(paths)} images under {data_dir}...")
    stats = [
        compute_image_stats(
            path=path,
            thumbnail_size=args.thumbnail_size,
            black_pixel_threshold=args.black_pixel_threshold,
            bright_pixel_threshold=args.bright_pixel_threshold,
            mean_threshold=args.mean_threshold,
            p95_threshold=args.p95_threshold,
            black_ratio_threshold=args.black_ratio_threshold,
            max_bright_ratio=args.max_bright_ratio,
        )
        for path in paths
    ]

    black_images = [item for item in stats if item.is_black]
    report_path = output_dir / timestamped_name("black_image_report", ".csv")
    candidates_path = output_dir / timestamped_name("black_image_candidates", ".csv")
    write_report(report_path, stats)
    write_report(candidates_path, black_images)

    sheet_path = output_dir / timestamped_name("black_image_contact_sheet", ".jpg")
    make_contact_sheet(
        black_images[: args.contact_sheet_limit],
        sheet_path,
        tile_size=180,
    )

    print_summary(stats, black_images)
    print(f"Full report: {report_path}")
    print(f"Candidate report: {candidates_path}")
    if black_images:
        print(f"Contact sheet: {sheet_path}")

    if args.action == "dry-run":
        print("Dry run only. No files were modified.")
        return

    if args.action == "delete":
        delete_images(black_images)
        print(f"Deleted {len(black_images)} black images.")
        return

    quarantine_dir = Path(args.quarantine_dir)
    move_to_quarantine(black_images, data_dir, quarantine_dir)
    print(f"Moved {len(black_images)} black images to {quarantine_dir}.")


def iter_image_paths(data_dir: Path, includes: Optional[Sequence[str]]) -> Iterable[Path]:
    include_roots: List[Path]
    if includes:
        include_roots = []
        for include in includes:
            root = Path(include)
            if not root.exists():
                root = data_dir / include
            if not root.exists():
                raise FileNotFoundError(f"Included folder not found: {include}")
            include_roots.append(root)
    else:
        include_roots = [data_dir]

    for root in include_roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield path


def compute_image_stats(
    path: Path,
    thumbnail_size: int,
    black_pixel_threshold: int,
    bright_pixel_threshold: int,
    mean_threshold: float,
    p95_threshold: float,
    black_ratio_threshold: float,
    max_bright_ratio: float,
) -> ImageStats:
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        gray.thumbnail((thumbnail_size, thumbnail_size))
        pixels = np.asarray(gray, dtype=np.uint8)

    mean = float(pixels.mean())
    std = float(pixels.std())
    p95 = float(np.percentile(pixels, 95))
    p99 = float(np.percentile(pixels, 99))
    black_ratio = float((pixels <= black_pixel_threshold).mean())
    bright_ratio = float((pixels >= bright_pixel_threshold).mean())

    is_black = (
        mean <= mean_threshold
        and p95 <= p95_threshold
        and black_ratio >= black_ratio_threshold
        and bright_ratio <= max_bright_ratio
    )

    return ImageStats(
        path=path,
        width=width,
        height=height,
        mean=mean,
        std=std,
        p95=p95,
        p99=p99,
        black_ratio=black_ratio,
        bright_ratio=bright_ratio,
        is_black=is_black,
    )


def write_report(path: Path, rows: Sequence[ImageStats]) -> None:
    columns = [
        "path",
        "is_black",
        "width",
        "height",
        "mean",
        "std",
        "p95",
        "p99",
        "black_ratio",
        "bright_ratio",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "path": str(item.path),
                    "is_black": int(item.is_black),
                    "width": item.width,
                    "height": item.height,
                    "mean": f"{item.mean:.4f}",
                    "std": f"{item.std:.4f}",
                    "p95": f"{item.p95:.4f}",
                    "p99": f"{item.p99:.4f}",
                    "black_ratio": f"{item.black_ratio:.6f}",
                    "bright_ratio": f"{item.bright_ratio:.6f}",
                }
            )


def make_contact_sheet(rows: Sequence[ImageStats], path: Path, tile_size: int) -> None:
    if not rows:
        return

    columns = 5
    label_height = 48
    rows_count = math.ceil(len(rows) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_size, rows_count * (tile_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for index, item in enumerate(rows):
        col = index % columns
        row = index // columns
        x = col * tile_size
        y = row * (tile_size + label_height)

        with Image.open(item.path) as image:
            image = image.convert("RGB")
            image.thumbnail((tile_size, tile_size))
            x_offset = x + (tile_size - image.width) // 2
            y_offset = y + (tile_size - image.height) // 2
            sheet.paste(image, (x_offset, y_offset))

        label = (
            f"mean={item.mean:.1f} p95={item.p95:.1f}\n"
            f"black={item.black_ratio:.2f}"
        )
        draw.text((x + 6, y + tile_size + 4), label, fill="black")

    sheet.save(path, quality=92)


def print_summary(stats: Sequence[ImageStats], black_images: Sequence[ImageStats]) -> None:
    means = np.asarray([item.mean for item in stats], dtype=float)
    p95s = np.asarray([item.p95 for item in stats], dtype=float)
    black_ratios = np.asarray([item.black_ratio for item in stats], dtype=float)

    print(f"Detected black images: {len(black_images)} / {len(stats)}")
    print(
        "Brightness mean percentiles: "
        f"min={means.min():.1f}, p1={np.percentile(means, 1):.1f}, "
        f"p5={np.percentile(means, 5):.1f}, median={np.median(means):.1f}"
    )
    print(
        "P95 brightness percentiles: "
        f"min={p95s.min():.1f}, p1={np.percentile(p95s, 1):.1f}, "
        f"p5={np.percentile(p95s, 5):.1f}, median={np.median(p95s):.1f}"
    )
    print(
        "Black-pixel ratio percentiles: "
        f"median={np.median(black_ratios):.3f}, "
        f"p95={np.percentile(black_ratios, 95):.3f}, "
        f"max={black_ratios.max():.3f}"
    )

    if black_images:
        print("Darkest candidates:")
        for item in sorted(black_images, key=lambda row: row.mean)[:10]:
            print(
                f"  mean={item.mean:.1f} p95={item.p95:.1f} "
                f"black={item.black_ratio:.3f} {item.path}"
            )


def delete_images(rows: Sequence[ImageStats]) -> None:
    for item in rows:
        item.path.unlink()


def move_to_quarantine(rows: Sequence[ImageStats], data_dir: Path, quarantine_dir: Path) -> None:
    for item in rows:
        relative = item.path.relative_to(data_dir)
        destination = quarantine_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item.path), str(destination))


def timestamped_name(prefix: str, suffix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}{suffix}"


if __name__ == "__main__":
    main()
