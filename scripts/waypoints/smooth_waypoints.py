#!/usr/bin/env python3
"""Smooth and map-constrain waypoint CSV to stay in hallway free space."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def load_map_yaml(yaml_path: Path):
    with yaml_path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    image_path = (yaml_path.parent / meta["image"]).resolve()
    resolution = float(meta["resolution"])
    origin_x = float(meta["origin"][0])
    origin_y = float(meta["origin"][1])
    negate = int(meta.get("negate", 0))
    occupied_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.25))
    return (
        image_path,
        resolution,
        origin_x,
        origin_y,
        negate,
        occupied_thresh,
        free_thresh,
    )


def world_to_pixel(x, y, resolution, origin_x, origin_y, image_h):
    px = (x - origin_x) / resolution
    py_from_bottom = (y - origin_y) / resolution
    py = (image_h - 1) - py_from_bottom
    return px, py


def pixel_to_world(px, py, resolution, origin_x, origin_y, image_h):
    x = px * resolution + origin_x
    py_from_bottom = (image_h - 1) - py
    y = py_from_bottom * resolution + origin_y
    return x, y


def occupancy_prob_from_pgm(gray, negate):
    gray = gray.astype(np.float32)
    if negate == 0:
        return (255.0 - gray) / 255.0
    return gray / 255.0


def build_free_mask(prob, free_thresh):
    # free if occupancy probability is below free threshold
    return prob < free_thresh


def integral_image(binary_img):
    integ = binary_img.astype(np.int32).cumsum(axis=0).cumsum(axis=1)
    # 1-pixel padding to simplify area queries
    padded = np.zeros((integ.shape[0] + 1, integ.shape[1] + 1), dtype=np.int32)
    padded[1:, 1:] = integ
    return padded


def window_sum(integ, x0, y0, x1, y1):
    # inclusive x0,y0,x1,y1
    return integ[y1 + 1, x1 + 1] - integ[y0, x1 + 1] - integ[y1 + 1, x0] + integ[y0, x0]


def build_safe_mask(free_mask, clearance_px):
    if clearance_px <= 0:
        return free_mask.copy()

    h, w = free_mask.shape
    integ = integral_image(free_mask)
    safe = np.zeros_like(free_mask, dtype=bool)
    side = 2 * clearance_px + 1
    area = side * side

    for y in range(clearance_px, h - clearance_px):
        y0 = y - clearance_px
        y1 = y + clearance_px
        for x in range(clearance_px, w - clearance_px):
            x0 = x - clearance_px
            x1 = x + clearance_px
            safe[y, x] = window_sum(integ, x0, y0, x1, y1) == area
    return safe


def decimate(points, min_spacing):
    if len(points) == 0:
        return points
    kept = [points[0]]
    last = points[0]
    for p in points[1:]:
        if np.linalg.norm(p - last) >= min_spacing:
            kept.append(p)
            last = p
    return np.asarray(kept, dtype=np.float64)


def moving_average(points, window):
    if window <= 1 or len(points) < window:
        return points.copy()
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(points, dtype=np.float64)
    for i in range(len(points)):
        out[i] = padded[i : i + window].mean(axis=0)
    return out


def resample_by_spacing(points, spacing):
    if len(points) <= 1:
        return points.copy()
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 1e-6:
        return points[:1].copy()
    n = max(int(total / spacing), 2)
    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=np.float64)
    out[:, 0] = np.interp(targets, s, points[:, 0])
    out[:, 1] = np.interp(targets, s, points[:, 1])
    return out


def arc_length(points):
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float64)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def corner_inward_pull(
    points,
    angle_deg_min,
    lead_m,
    pull_m,
    also_after_m=0.0,
):
    """Pull path toward the inside of sharp turns (tighter line, earlier turn-in).

    At each vertex with turning angle >= angle_deg_min, the "inside" direction is
    approximately normalize(u_in + u_out) where u_in/u_out are unit directions along
    the path into and out of the vertex. We apply a tapering offset to points up to
    `lead_m` *before* the corner (and optionally `also_after_m` after) so the car
    commits to the turn earlier (hug inner wall in hallways).
    """
    n = len(points)
    if n < 3 or pull_m <= 0:
        return points.copy()

    s = arc_length(points)
    deltas = np.zeros_like(points)

    for i in range(1, n - 1):
        v0 = points[i] - points[i - 1]
        v1 = points[i + 1] - points[i]
        la = np.linalg.norm(v0)
        lb = np.linalg.norm(v1)
        if la < 1e-9 or lb < 1e-9:
            continue
        a0 = v0 / la
        a1 = v1 / lb
        c = float(np.clip(np.dot(a0, a1), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(c)))
        if angle < angle_deg_min:
            continue

        inward = a0 + a1
        ni = np.linalg.norm(inward)
        if ni < 1e-9:
            continue
        inward = inward / ni

        # Before corner: strongest at the vertex, zero at lead_m back along path
        for j in range(n):
            if s[j] <= s[i] and s[i] - s[j] <= lead_m:
                d = s[i] - s[j]
                w = 1.0 - (d / lead_m) if lead_m > 0 else 1.0
                deltas[j] += inward * (pull_m * w)

        # After corner: optional short taper (helps if path overshoots the straight)
        if also_after_m > 0:
            for j in range(n):
                if s[j] > s[i] and s[j] - s[i] <= also_after_m:
                    d = s[j] - s[i]
                    w = 1.0 - (d / also_after_m)
                    deltas[j] += inward * (pull_m * 0.35 * w)

    return points + deltas


def nearest_safe_pixel(px, py, safe_mask, max_radius):
    h, w = safe_mask.shape
    cx = int(round(px))
    cy = int(round(py))

    if 0 <= cx < w and 0 <= cy < h and safe_mask[cy, cx]:
        return cx, cy

    for r in range(1, max_radius + 1):
        x0 = max(0, cx - r)
        x1 = min(w - 1, cx + r)
        y0 = max(0, cy - r)
        y1 = min(h - 1, cy + r)

        best = None
        best_d2 = None

        for x in range(x0, x1 + 1):
            for y in (y0, y1):
                if safe_mask[y, x]:
                    d2 = (x - cx) ** 2 + (y - cy) ** 2
                    if best_d2 is None or d2 < best_d2:
                        best = (x, y)
                        best_d2 = d2
        for y in range(y0 + 1, y1):
            for x in (x0, x1):
                if safe_mask[y, x]:
                    d2 = (x - cx) ** 2 + (y - cy) ** 2
                    if best_d2 is None or d2 < best_d2:
                        best = (x, y)
                        best_d2 = d2
        if best is not None:
            return best

    # If no safe pixel found nearby, clamp to image bounds.
    return np.clip(cx, 0, w - 1), np.clip(cy, 0, h - 1)


def save_csv(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for p in points:
            writer.writerow([f"{p[0]:.8f}", f"{p[1]:.8f}"])


def main():
    parser = argparse.ArgumentParser(description="Smooth waypoints with map constraints.")
    parser.add_argument("--map-yaml", default="data/maps/levine/race2_map.yaml")
    parser.add_argument(
        "--input",
        default="data/waypoints/race2_waypoints_full_smoothed.csv",
        help="Input waypoint CSV",
    )
    parser.add_argument(
        "--output",
        default="outputs/waypoints/race2_waypoints_smoothed.csv",
        help="Output waypoint CSV",
    )
    parser.add_argument("--min-spacing", type=float, default=0.12)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--resample-spacing", type=float, default=0.20)
    parser.add_argument(
        "--clearance-m",
        type=float,
        default=0.15,
        help="Required wall clearance in meters",
    )
    parser.add_argument(
        "--search-radius-px",
        type=int,
        default=20,
        help="Max snap distance (pixels) to nearest safe point",
    )
    parser.add_argument(
        "--corner-pull-m",
        type=float,
        default=0.12,
        help="Shift path toward inner side of sharp corners (m); 0 disables",
    )
    parser.add_argument(
        "--corner-angle-deg",
        type=float,
        default=28.0,
        help="Apply corner pull where turn angle exceeds this (degrees)",
    )
    parser.add_argument(
        "--corner-lead-m",
        type=float,
        default=1.2,
        help="Along-path distance before corner to taper pull (meters)",
    )
    parser.add_argument(
        "--corner-after-m",
        type=float,
        default=0.0,
        help="Optional taper after corner (m); 0 = only pull before corner",
    )
    args = parser.parse_args()

    map_yaml = Path(args.map_yaml).resolve()
    input_csv = Path(args.input).resolve()
    output_csv = Path(args.output).resolve()

    (
        image_path,
        resolution,
        origin_x,
        origin_y,
        negate,
        _occupied_thresh,
        free_thresh,
    ) = load_map_yaml(map_yaml)

    gray = np.array(Image.open(image_path).convert("L"))
    h, w = gray.shape

    raw = np.loadtxt(input_csv, delimiter=",", skiprows=1)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("Input CSV must have at least two columns: x,y")
    points = raw[:, :2].astype(np.float64)

    # 1) Remove near duplicates/noise.
    points = decimate(points, args.min_spacing)
    # 2) Smooth path geometry.
    points = moving_average(points, args.smooth_window)
    # 3) Uniform spacing for stable tracking.
    points = resample_by_spacing(points, args.resample_spacing)

    # 3b) Tighten sharp corners (e.g. upper-left): start turn earlier, hug inner wall.
    if args.corner_pull_m > 0:
        points = corner_inward_pull(
            points,
            angle_deg_min=args.corner_angle_deg,
            lead_m=args.corner_lead_m,
            pull_m=args.corner_pull_m,
            also_after_m=args.corner_after_m,
        )

    # 4) Build safe mask from map and push points away from walls.
    prob = occupancy_prob_from_pgm(gray, negate)
    free_mask = build_free_mask(prob, free_thresh)
    clearance_px = max(0, int(round(args.clearance_m / resolution)))
    safe_mask = build_safe_mask(free_mask, clearance_px)

    px, py = world_to_pixel(points[:, 0], points[:, 1], resolution, origin_x, origin_y, h)
    snapped = np.zeros_like(points)
    moved = 0
    for i in range(len(points)):
        sx, sy = nearest_safe_pixel(px[i], py[i], safe_mask, args.search_radius_px)
        if (int(round(px[i])) != sx) or (int(round(py[i])) != sy):
            moved += 1
        wx, wy = pixel_to_world(sx, sy, resolution, origin_x, origin_y, h)
        snapped[i] = [wx, wy]

    save_csv(output_csv, snapped)

    print(f"Input waypoints: {len(raw)}")
    print(f"Output waypoints: {len(snapped)}")
    print(f"Map: {image_path} ({w}x{h}), resolution={resolution}")
    print(f"Clearance: {args.clearance_m:.3f} m ({clearance_px} px)")
    print(f"Moved-to-safe points: {moved}/{len(snapped)}")
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
