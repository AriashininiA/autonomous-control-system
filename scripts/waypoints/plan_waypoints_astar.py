#!/usr/bin/env python3
"""
Connect waypoint keyframes with paths that stay inside map free space (hallways).

For each consecutive pair (including loop closure), if the straight segment crosses
non-free cells, run A* on an 8-connected grid cropped around the segment.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def _load_smooth_waypoints_module():
    here = Path(__file__).resolve().parent / "smooth_waypoints.py"
    spec = importlib.util.spec_from_file_location("smooth_waypoints", here)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def is_closed_loop(points: np.ndarray, tol: float = 0.35) -> bool:
    if len(points) < 3:
        return False
    return float(np.linalg.norm(points[0] - points[-1])) < tol


def nearest_free_cell(
    cx: int,
    cy: int,
    walkable: np.ndarray,
    max_search: int = 80,
) -> tuple[int, int] | None:
    h, w = walkable.shape
    if 0 <= cx < w and 0 <= cy < h and walkable[cy, cx]:
        return cx, cy
    for r in range(1, max_search + 1):
        x0, x1 = max(0, cx - r), min(w - 1, cx + r)
        y0, y1 = max(0, cy - r), min(h - 1, cy + r)
        for x in range(x0, x1 + 1):
            for y in (y0, y1):
                if walkable[y, x]:
                    return x, y
        for y in range(y0 + 1, y1):
            for x in (x0, x1):
                if walkable[y, x]:
                    return x, y
    return None


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer grid cells along a line (inclusive)."""
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def line_crosses_nonfree(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    walkable: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    image_h: int,
    sw,
) -> bool:
    """True if segment crosses a non-walkable map cell (pixel-accurate)."""
    pxa, pya = sw.world_to_pixel(
        np.array([x0, x1]),
        np.array([y0, y1]),
        resolution,
        origin_x,
        origin_y,
        image_h,
    )
    sx, sy = int(round(float(pxa[0]))), int(round(float(pya[0])))
    gx, gy = int(round(float(pxa[1]))), int(round(float(pya[1])))
    h, w = walkable.shape
    for cx, cy in bresenham_line(sx, sy, gx, gy):
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return True
        if not walkable[cy, cx]:
            return True
    return False


NEIGH = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, np.sqrt(2.0)),
    (1, -1, np.sqrt(2.0)),
    (-1, 1, np.sqrt(2.0)),
    (-1, -1, np.sqrt(2.0)),
]


def astar_crop(
    walkable_full: np.ndarray,
    sx: int,
    sy: int,
    gx: int,
    gy: int,
    margin: int,
) -> list[tuple[int, int]] | None:
    h, w = walkable_full.shape
    x0 = max(0, min(sx, gx) - margin)
    x1 = min(w - 1, max(sx, gx) + margin)
    y0 = max(0, min(sy, gy) - margin)
    y1 = min(h - 1, max(sy, gy) + margin)
    if x0 > x1 or y0 > y1:
        return None

    sub = walkable_full[y0 : y1 + 1, x0 : x1 + 1]
    sh, sw_ = sub.shape
    lsx, lsy = sx - x0, sy - y0
    lgx, lgy = gx - x0, gy - y0

    if not (0 <= lsx < sw_ and 0 <= lsy < sh and sub[lsy, lsx]):
        nf = nearest_free_cell(lsx, lsy, sub, max_search=min(margin, 60))
        if nf is None:
            return None
        lsx, lsy = nf
    if not (0 <= lgx < sw_ and 0 <= lgy < sh and sub[lgy, lgx]):
        nf = nearest_free_cell(lgx, lgy, sub, max_search=min(margin, 60))
        if nf is None:
            return None
        lgx, lgy = nf

    def hfun(x: int, y: int) -> float:
        return float(np.hypot(lgx - x, lgy - y))

    open_heap: list[tuple[float, float, int, int]] = []
    gscore: dict[tuple[int, int], float] = {}
    came: dict[tuple[int, int], tuple[int, int]] = {}

    heapq.heappush(open_heap, (hfun(lsx, lsy), 0.0, lsx, lsy))
    gscore[(lsx, lsy)] = 0.0

    while open_heap:
        _, g, x, y = heapq.heappop(open_heap)
        if (x, y) == (lgx, lgy):
            path = [(x, y)]
            while (x, y) in came:
                x, y = came[(x, y)]
                path.append((x, y))
            path.reverse()
            return [(p[0] + x0, p[1] + y0) for p in path]

        if gscore.get((x, y), 1e30) < g - 1e-9:
            continue

        for dx, dy, cost in NEIGH:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= sw_ or ny >= sh:
                continue
            if not sub[ny, nx]:
                continue
            ng = g + cost
            if ng < gscore.get((nx, ny), 1e30):
                gscore[(nx, ny)] = ng
                came[(nx, ny)] = (x, y)
                heapq.heappush(open_heap, (ng + hfun(nx, ny), ng, nx, ny))

    return None


def decimate(points: np.ndarray, min_spacing: float) -> np.ndarray:
    if len(points) == 0:
        return points
    kept = [points[0]]
    last = points[0]
    for p in points[1:]:
        if np.linalg.norm(p - last) >= min_spacing:
            kept.append(p)
            last = p
    return np.asarray(kept, dtype=np.float64)


def find_first_bad_edge(
    pts: np.ndarray,
    closed: bool,
    walkable: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    image_h: int,
    sw,
) -> int | None:
    """Return index i such that edge (i,i+1) is bad, or -1 for closing edge (last->0)."""
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if line_crosses_nonfree(
            a[0], a[1], b[0], b[1], walkable, resolution, origin_x, origin_y, image_h, sw
        ):
            return i
    if closed and len(pts) >= 2:
        a, b = pts[-1], pts[0]
        if line_crosses_nonfree(
            a[0], a[1], b[0], b[1], walkable, resolution, origin_x, origin_y, image_h, sw
        ):
            return -1
    return None


def splice_astar_edge(
    pts: np.ndarray,
    edge_idx: int,
    closed: bool,
    walkable: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    image_h: int,
    astar_margin_px: int,
    sample_spacing_m: float,
    sw,
) -> np.ndarray | None:
    if edge_idx == -1:
        xa, ya = pts[-1]
        xb, yb = pts[0]
    else:
        xa, ya = pts[edge_idx]
        xb, yb = pts[edge_idx + 1]

    pxa, pya = sw.world_to_pixel(
        np.array([xa]), np.array([ya]), resolution, origin_x, origin_y, image_h
    )
    pxb, pyb = sw.world_to_pixel(
        np.array([xb]), np.array([yb]), resolution, origin_x, origin_y, image_h
    )
    sx, sy = int(round(float(pxa[0]))), int(round(float(pya[0])))
    gx, gy = int(round(float(pxb[0]))), int(round(float(pyb[0])))
    px_path = astar_crop(walkable, sx, sy, gx, gy, astar_margin_px)
    if px_path is None:
        px_path = astar_crop(walkable, sx, sy, gx, gy, min(astar_margin_px * 2, 450))
    if px_path is None:
        return None
    # Use every grid step along A* so chords stay inside the hallway
    raw_world = []
    for cx, cy in px_path:
        wx, wy = sw.pixel_to_world(
            float(cx), float(cy), resolution, origin_x, origin_y, image_h
        )
        raw_world.append([wx, wy])
    sub = np.asarray(raw_world, dtype=np.float64)
    sub = decimate(sub, max(sample_spacing_m * 0.55, 0.035))
    if len(sub) < 2:
        return pts
    a = np.array([xa, ya])
    b = np.array([xb, yb])
    tol = 0.04
    mid_rows: list[np.ndarray] = []
    for row in sub:
        if np.linalg.norm(row - a) < tol or np.linalg.norm(row - b) < tol:
            continue
        mid_rows.append(row)
    if not mid_rows:
        # keep at least one point away from endpoints
        mid_rows = [sub[len(sub) // 2]]
    mid = np.asarray(mid_rows, dtype=np.float64)

    if edge_idx == -1:
        return np.vstack([pts, mid])
    return np.vstack([pts[: edge_idx + 1], mid, pts[edge_idx + 1 :]])


def repair_all_shortcuts(
    pts: np.ndarray,
    closed: bool,
    walkable: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    image_h: int,
    astar_margin_px: int,
    sample_spacing_m: float,
    sw,
    max_iter: int = 400,
) -> tuple[np.ndarray, int]:
    """Iteratively replace chords that cross obstacles with A* corridor paths."""
    cur = pts.copy()
    repairs = 0
    stale = 0
    for _ in range(max_iter):
        bi = find_first_bad_edge(
            cur, closed, walkable, resolution, origin_x, origin_y, image_h, sw
        )
        if bi is None:
            break
        nxt = splice_astar_edge(
            cur,
            bi,
            closed,
            walkable,
            resolution,
            origin_x,
            origin_y,
            image_h,
            astar_margin_px,
            sample_spacing_m,
            sw,
        )
        if nxt is None:
            break
        if len(nxt) <= len(cur):
            stale += 1
            if stale > 25:
                break
            continue
        stale = 0
        cur = nxt
        repairs += 1
    return cur, repairs


def save_csv(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for p in points:
            w.writerow([f"{p[0]:.8f}", f"{p[1]:.8f}"])


def load_csv(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            rows.append([x, y])
    if len(rows) < 2:
        raise ValueError(f"Need at least 2 numeric x,y rows in {path}")
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan hallway-constrained paths between waypoint keyframes (A*)."
    )
    parser.add_argument("--map-yaml", default="data/maps/levine/race2_map.yaml")
    parser.add_argument(
        "--input",
        default="data/waypoints/race2_waypoints_full_smoothed.csv",
        help="Keyframe CSV (x,y); used as segment endpoints",
    )
    parser.add_argument(
        "--output",
        default="outputs/waypoints/race2_waypoints_astar_repaired.csv",
    )
    parser.add_argument(
        "--keyframes",
        type=int,
        default=0,
        help="If >0, subsample input to this many roughly-even keyframes (0=all rows)",
    )
    parser.add_argument(
        "--astar-margin-px",
        type=int,
        default=140,
        help="Crop margin around segment for A* (pixels)",
    )
    parser.add_argument(
        "--sample-spacing-m",
        type=float,
        default=0.14,
        help="Output spacing along planned path (m)",
    )
    parser.add_argument(
        "--output-spacing-m",
        type=float,
        default=0.15,
        help="Final decimation (m)",
    )
    parser.add_argument("--loop-tol", type=float, default=0.35)
    parser.add_argument(
        "--close-loop",
        action="store_true",
        help="Append first point at end if not already closed (loop tracks)",
    )
    args = parser.parse_args()

    sw = _load_smooth_waypoints_module()
    map_yaml = Path(args.map_yaml).resolve()
    in_csv = Path(args.input).resolve()
    out_csv = Path(args.output).resolve()

    (
        image_path,
        resolution,
        origin_x,
        origin_y,
        negate,
        _,
        free_thresh,
    ) = sw.load_map_yaml(map_yaml)
    gray = np.array(Image.open(image_path).convert("L"))
    h, w = gray.shape
    prob = sw.occupancy_prob_from_pgm(gray, negate)
    free_mask = sw.build_free_mask(prob, free_thresh)
    walkable = free_mask.copy()

    pts = load_csv(in_csv)
    closed = is_closed_loop(pts, tol=args.loop_tol)
    if args.close_loop and not closed:
        pts = np.vstack([pts, pts[0:1]])
        closed = True
    if closed and np.linalg.norm(pts[0] - pts[-1]) < 1e-4:
        key = pts[:-1].copy()
    else:
        key = pts.copy()

    if args.keyframes > 0 and len(key) > args.keyframes:
        idx = np.linspace(0, len(key) - 1, args.keyframes, dtype=int)
        key = key[np.unique(idx)]

    n = len(key)
    if n < 2:
        raise SystemExit("Need at least 2 keyframe points")

    all_segments: list[np.ndarray] = []

    def seg_indices():
        for i in range(n - 1):
            yield i, i + 1
        if closed:
            yield n - 1, 0

    astar_used = 0
    for ia, ib in seg_indices():
        xa, ya = key[ia]
        xb, yb = key[ib]

        if not line_crosses_nonfree(
            xa,
            ya,
            xb,
            yb,
            walkable,
            resolution,
            origin_x,
            origin_y,
            h,
            sw,
        ):
            px_path = None
        else:
            pxa, pya = sw.world_to_pixel(
                np.array([xa]),
                np.array([ya]),
                resolution,
                origin_x,
                origin_y,
                h,
            )
            pxb, pyb = sw.world_to_pixel(
                np.array([xb]),
                np.array([yb]),
                resolution,
                origin_x,
                origin_y,
                h,
            )
            sx, sy = int(round(float(pxa[0]))), int(round(float(pya[0])))
            gx, gy = int(round(float(pxb[0]))), int(round(float(pyb[0])))
            px_path = astar_crop(walkable, sx, sy, gx, gy, args.astar_margin_px)
            if px_path is None:
                px_path = astar_crop(
                    walkable, sx, sy, gx, gy, min(args.astar_margin_px * 2, 400)
                )
            if px_path is None:
                print(
                    f"Warning: A* failed for segment {ia}->{ib}, "
                    f"falling back to dense straight line (may cut walls)."
                )
            else:
                astar_used += 1

        if px_path is None:
            # Dense straight world samples (still bad if blocked; user sees warning)
            d = float(np.hypot(xb - xa, yb - ya))
            ns = max(2, int(np.ceil(d / args.sample_spacing_m)))
            t = np.linspace(0, 1, ns)
            seg_pts = np.column_stack([(1 - t) * xa + t * xb, (1 - t) * ya + t * yb])
        else:
            raw_world = []
            for cx, cy in px_path:
                wx, wy = sw.pixel_to_world(
                    float(cx), float(cy), resolution, origin_x, origin_y, h
                )
                raw_world.append([wx, wy])
            seg_pts = np.asarray(raw_world, dtype=np.float64)
            seg_pts = decimate(seg_pts, max(args.sample_spacing_m * 0.55, 0.035))

        if len(all_segments) > 0 and len(seg_pts) > 0:
            if np.linalg.norm(all_segments[-1][-1] - seg_pts[0]) < 1e-3:
                seg_pts = seg_pts[1:]
        if len(seg_pts) > 0:
            all_segments.append(seg_pts)

    if not all_segments:
        raise SystemExit("Empty path")

    merged = np.vstack(all_segments)
    merged = decimate(merged, args.output_spacing_m)

    if closed and len(merged) >= 2:
        if np.linalg.norm(merged[0] - merged[-1]) < 0.08:
            merged_body = merged[:-1].copy()
        else:
            merged_body = merged.copy()
    else:
        merged_body = merged.copy()

    merged_body, shortcut_repairs = repair_all_shortcuts(
        merged_body,
        closed,
        walkable,
        resolution,
        origin_x,
        origin_y,
        h,
        args.astar_margin_px,
        min(args.sample_spacing_m, 0.05),
        sw,
    )

    if closed:
        merged = np.vstack([merged_body, merged_body[0:1]])
    else:
        merged = merged_body

    save_csv(out_csv, merged)

    bad_edges = 0
    for i in range(len(merged) - 1):
        if np.linalg.norm(merged[i + 1] - merged[i]) < 1e-6:
            continue
        xa, ya = merged[i]
        xb, yb = merged[i + 1]
        if line_crosses_nonfree(
            xa,
            ya,
            xb,
            yb,
            walkable,
            resolution,
            origin_x,
            origin_y,
            h,
            sw,
        ):
            bad_edges += 1

    print(f"Keyframes used: {n} (closed loop: {closed})")
    print(f"A* replans (keyframes): {astar_used}")
    print(f"Shortcut repairs (output): {shortcut_repairs}")
    print(f"Output points: {len(merged)}")
    print(f"Output edges crossing non-free (want 0): {bad_edges}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
