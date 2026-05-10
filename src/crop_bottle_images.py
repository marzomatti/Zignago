#!/usr/bin/env python3
"""Crop glass bottle BMP images into the smaller inspection region.

The crop window is selected from the filename suffix:

    *_80_2_C.bmp -> (0, 375) to (600, 725)
    *_26_2_C.bmp -> (0, 400) to (600, 750)
    *_28_2_C.bmp -> (0, 300) to (600, 650)

Coordinates use the standard image convention: origin at the top-left pixel,
x increasing to the right, y increasing downward. PIL crop boxes use
(left, upper, right, lower), with right/lower exclusive, so each crop is exactly
600 pixels wide and 350 pixels high.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CROP_RULES: Dict[str, Tuple[int, int, int, int]] = {
    "80_2_C.bmp": (0, 375, 600, 725),
    "26_2_C.bmp": (0, 400, 600, 750),
    "28_2_C.bmp": (0, 300, 600, 650),
}


@dataclass
class CropResult:
    source: Path
    destination: Path
    status: str
    rule: str
    crop_box: Optional[Tuple[int, int, int, int]]
    source_width: Optional[int]
    source_height: Optional[int]
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop bottle images from an input parent directory into an output child directory."
    )
    parser.add_argument(
        "parent_dir",
        help="Parent directory containing original images. The scan is recursive.",
    )
    parser.add_argument(
        "child_dir",
        help="Output directory where cropped images will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already existing cropped images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and report what would be cropped without writing images.",
    )
    parser.add_argument(
        "--fail-on-unknown",
        action="store_true",
        help="Fail if an image does not match one of the known suffix rules.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional CSV report path. Defaults to <child_dir>/crop_report.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent_dir = Path(args.parent_dir).expanduser().resolve()
    child_dir = Path(args.child_dir).expanduser().resolve()

    if not parent_dir.exists() or not parent_dir.is_dir():
        raise FileNotFoundError(f"Parent directory not found: {parent_dir}")
    if parent_dir == child_dir:
        raise ValueError("parent_dir and child_dir must be different directories.")
    if parent_dir in child_dir.parents:
        # This is fine for a child output folder, but avoid scanning it later by
        # materializing the input file list before writing anything.
        pass

    image_paths = list(iter_image_paths(parent_dir, child_dir))
    if not image_paths:
        raise SystemExit(f"No image files found under {parent_dir}")

    if not args.dry_run:
        child_dir.mkdir(parents=True, exist_ok=True)

    results: List[CropResult] = []
    for source in image_paths:
        destination = child_dir / source.relative_to(parent_dir)
        result = crop_one(
            source=source,
            destination=destination,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            fail_on_unknown=args.fail_on_unknown,
        )
        results.append(result)

    report_path = Path(args.report) if args.report else child_dir / "crop_report.csv"
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report_path, results)
    print_summary(results, report_path, args.dry_run)

    failed = [result for result in results if result.status == "failed"]
    unknown = [result for result in results if result.status == "unknown_suffix"]
    if failed or (args.fail_on_unknown and unknown):
        raise SystemExit(1)


def iter_image_paths(parent_dir: Path, child_dir: Path) -> Iterable[Path]:
    for path in sorted(parent_dir.rglob("*")):
        if not path.is_file():
            continue
        if child_dir == path or child_dir in path.parents:
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def crop_one(
    source: Path,
    destination: Path,
    overwrite: bool,
    dry_run: bool,
    fail_on_unknown: bool,
) -> CropResult:
    rule_name, crop_box = find_crop_rule(source.name)
    if crop_box is None:
        status = "failed" if fail_on_unknown else "unknown_suffix"
        return CropResult(
            source=source,
            destination=destination,
            status=status,
            rule="",
            crop_box=None,
            source_width=None,
            source_height=None,
            message="No crop rule matched this filename suffix.",
        )

    try:
        with Image.open(source) as image:
            width, height = image.size
            left, upper, right, lower = crop_box
            if right > width or lower > height:
                return CropResult(
                    source=source,
                    destination=destination,
                    status="failed",
                    rule=rule_name,
                    crop_box=crop_box,
                    source_width=width,
                    source_height=height,
                    message=(
                        f"Crop box {crop_box} exceeds image bounds "
                        f"{width}x{height}."
                    ),
                )

            if destination.exists() and not overwrite:
                return CropResult(
                    source=source,
                    destination=destination,
                    status="skipped_exists",
                    rule=rule_name,
                    crop_box=crop_box,
                    source_width=width,
                    source_height=height,
                    message="Destination exists. Use --overwrite to replace it.",
                )

            if dry_run:
                return CropResult(
                    source=source,
                    destination=destination,
                    status="would_crop",
                    rule=rule_name,
                    crop_box=crop_box,
                    source_width=width,
                    source_height=height,
                    message="Dry run; image was not written.",
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            cropped = image.crop(crop_box)
            cropped.save(destination, format="BMP")

        return CropResult(
            source=source,
            destination=destination,
            status="cropped",
            rule=rule_name,
            crop_box=crop_box,
            source_width=width,
            source_height=height,
            message="",
        )
    except Exception as exc:
        return CropResult(
            source=source,
            destination=destination,
            status="failed",
            rule=rule_name,
            crop_box=crop_box,
            source_width=None,
            source_height=None,
            message=str(exc),
        )


def find_crop_rule(filename: str) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
    for suffix, crop_box in CROP_RULES.items():
        if filename.endswith(suffix):
            return suffix, crop_box
    return "", None


def write_report(path: Path, results: List[CropResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "destination",
                "status",
                "rule",
                "crop_box",
                "source_width",
                "source_height",
                "message",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "source": str(result.source),
                    "destination": str(result.destination),
                    "status": result.status,
                    "rule": result.rule,
                    "crop_box": "" if result.crop_box is None else result.crop_box,
                    "source_width": result.source_width,
                    "source_height": result.source_height,
                    "message": result.message,
                }
            )


def print_summary(results: List[CropResult], report_path: Path, dry_run: bool) -> None:
    counts: Dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    print(f"Scanned images: {len(results)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    print(f"Report: {report_path}")
    if dry_run:
        print("Dry run only. No cropped images were written.")


if __name__ == "__main__":
    main()
