from __future__ import annotations

import numpy as np


def nearest_point(point: np.ndarray, trajectory: np.ndarray):
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    l2s = np.maximum(l2s, 1e-12)
    dots = np.sum((point - trajectory[:-1, :]) * diffs, axis=1)
    t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.linalg.norm(point - projections, axis=1)
    idx = int(np.argmin(dists))
    return projections[idx], float(dists[idx]), float(t[idx]), idx


def calc_interpolated_ref_trajectory(x, y, cx, cy, cv, cyaw, dt, horizon):
    ncourse = len(cx)
    if ncourse < 2:
        raise ValueError("MPC requires at least two waypoints")

    dl = float(np.hypot(cx[1] - cx[0], cy[1] - cy[0]))
    dl = max(dl, 1e-6)
    _, _, t_current, ind_current = nearest_point(np.array([x, y]), np.column_stack((cx, cy)))

    t_list = np.zeros(horizon + 1)
    t_list[0] = t_current
    ind_next = (ind_current + 1) % ncourse
    current_speed = (1.0 - t_current) * cv[ind_current] + t_current * cv[ind_next]

    for i in range(1, horizon + 1):
        t_list[i] = t_list[i - 1] + (current_speed * dt) / dl
        t_frac = t_list[i] % 1.0
        seg_idx = (int(t_list[i]) + ind_current) % ncourse
        seg_next = (seg_idx + 1) % ncourse
        current_speed = (1.0 - t_frac) * cv[seg_idx] + t_frac * cv[seg_next]

    ind_list = (np.floor(t_list).astype(np.int64) + ind_current) % ncourse
    t_frac_list = t_list % 1.0
    ref_traj = np.zeros((4, horizon + 1))

    for i in range(horizon + 1):
        idx = ind_list[i]
        idx_next = (idx + 1) % ncourse
        t_frac = t_frac_list[i]
        ref_traj[0, i] = (1.0 - t_frac) * cx[idx] + t_frac * cx[idx_next]
        ref_traj[1, i] = (1.0 - t_frac) * cy[idx] + t_frac * cy[idx_next]
        ref_traj[2, i] = (1.0 - t_frac) * cv[idx] + t_frac * cv[idx_next]
        yaw0 = cyaw[idx]
        yaw1 = cyaw[idx_next]
        yaw_diff = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))
        ref_traj[3, i] = yaw0 + t_frac * yaw_diff

    return ref_traj

