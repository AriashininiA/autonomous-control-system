#!/usr/bin/env python3
"""Overlay waypoint CSV coordinates onto a ROS occupancy map image."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


def load_map_metadata(yaml_path: Path) -> tuple[Path, float, float, float]:
    with yaml_path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    map_image = yaml_path.parent / meta["image"]
    resolution = float(meta["resolution"])
    origin_x = float(meta["origin"][0])
    origin_y = float(meta["origin"][1])
    return map_image, resolution, origin_x, origin_y


def world_to_pixel(
    x: np.ndarray,
    y: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Map frame uses bottom-left origin; image coordinates use top-left origin.
    px = (x - origin_x) / resolution
    py_from_bottom = (y - origin_y) / resolution
    py = (image_height - 1) - py_from_bottom
    return px, py


def _overlay_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay waypoints over a ROS map using map YAML metadata."
    )
    parser.add_argument(
        "--map-yaml",
        default="data/maps/levine/race2_map.yaml",
        help="Path to map YAML file",
    )
    parser.add_argument(
        "--waypoints",
        default="data/waypoints/race2_waypoints_full_smoothed.csv",
        help="Path to waypoint CSV with x,y columns",
    )
    parser.add_argument(
        "--save",
        default="outputs/waypoints/waypoints_overlay.png",
        help="Path to save overlay image",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open preview window; useful for SSH/headless runs",
    )
    parser.add_argument(
        "--no-stamp",
        action="store_true",
        help="Do not draw waypoint hash + timestamp (stamp proves the PNG matches current CSV)",
    )
    args = parser.parse_args()

    map_yaml = Path(args.map_yaml).resolve()
    waypoints_csv = Path(args.waypoints).resolve()
    out_path = Path(args.save).resolve()

    map_image_path, resolution, origin_x, origin_y = load_map_metadata(map_yaml)
    base_map = Image.open(map_image_path).convert("RGB")
    map_image = base_map.copy()
    waypoints = np.loadtxt(waypoints_csv, delimiter=",", skiprows=1)

    if waypoints.ndim != 2 or waypoints.shape[1] < 2:
        raise ValueError("Waypoint CSV must have at least two columns: x,y")

    x = waypoints[:, 0]
    y = waypoints[:, 1]

    width, height = map_image.size
    px, py = world_to_pixel(x, y, resolution, origin_x, origin_y, height)

    in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)

    draw = ImageDraw.Draw(map_image)
    # Cyan polyline reads clearly on gray map; red alone can look similar across small edits.
    line_rgb = (0, 200, 255)
    dot_rgb = (255, 40, 40)
    for i in range(len(px) - 1):
        x0, y0, x1, y1 = px[i], py[i], px[i + 1], py[i + 1]
        if (
            0 <= x0 < width
            and 0 <= y0 < height
            and 0 <= x1 < width
            and 0 <= y1 < height
        ):
            draw.line(
                (float(x0), float(y0), float(x1), float(y1)),
                fill=line_rgb,
                width=3,
            )
    for pxi, pyi in zip(px, py):
        if 0 <= pxi < width and 0 <= pyi < height:
            cx = int(round(pxi))
            cy = int(round(pyi))
            r = 2
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=dot_rgb)

    total = len(px)
    if not args.no_stamp:
        font = _overlay_font(13)
        blob = np.ascontiguousarray(np.column_stack([x, y]), dtype=np.float64).tobytes()
        whash = hashlib.sha256(blob).hexdigest()[:12]
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC %Y-%m-%d")
        label = (
            f"{waypoints_csv.name}  |  n={total}  |  xy_sha256={whash}...  |  {ts}"
        )
        margin = 6
        bar_h = 26
        draw.rectangle(
            (0, height - bar_h - margin, width, height - margin),
            fill=(18, 18, 18),
            outline=(255, 200, 0),
            width=1,
        )
        draw.text(
            (margin + 2, height - bar_h - margin + 4),
            label,
            fill=(255, 230, 80),
            font=font,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    map_image.save(out_path, format="PNG")
    if not args.no_show:
        map_image.show()

    inside = int(np.sum(in_bounds))
    print(f"Map image: {map_image_path}")
    print(f"Waypoint file: {waypoints_csv}")
    print(
        "Map metadata:"
        f" resolution={resolution}, origin=({origin_x}, {origin_y}), size=({width}, {height})"
    )
    print(f"In-bounds waypoints: {inside}/{total}")
    blob = np.ascontiguousarray(np.column_stack([x, y]), dtype=np.float64).tobytes()
    print(f"Waypoint xy SHA256 (first 16): {hashlib.sha256(blob).hexdigest()[:16]}...")
    print(f"Saved overlay image to: {out_path}")
    print(f"PNG file SHA256 (first 16): {hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]}...")


if __name__ == "__main__":
    main()
