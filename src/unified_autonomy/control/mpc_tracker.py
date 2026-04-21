from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import cvxpy
    from scipy.sparse import block_diag, csc_matrix
except ModuleNotFoundError:  # pragma: no cover - depends on optional MPC deps
    cvxpy = None
    block_diag = None
    csc_matrix = None

from unified_autonomy.control.mpc_utils import calc_interpolated_ref_trajectory
from unified_autonomy.interfaces import VehicleState


@dataclass
class MPCConfig:
    nx: int = 4
    nu: int = 2
    horizon: int = 8
    dt_s: float = 0.1
    wheelbase_m: float = 0.33
    min_steer_rad: float = -0.4189
    max_steer_rad: float = 0.4189
    max_dsteer_radps: float = float(np.deg2rad(180.0))
    max_speed_mps: float = 6.0
    min_speed_mps: float = 0.0
    max_accel_mps2: float = 3.0
    steering_sign: float = 1.0
    speed_track_gain: float = 0.65
    input_cost: np.ndarray = field(default_factory=lambda: np.diag([0.01, 3.0]))
    input_rate_cost: np.ndarray = field(default_factory=lambda: np.diag([0.01, 8.0]))
    state_cost: np.ndarray = field(default_factory=lambda: np.diag([15.0, 15.0, 5.5, 14.0]))
    final_state_cost: np.ndarray = field(default_factory=lambda: np.diag([15.0, 15.0, 5.5, 14.0]))


@dataclass
class MPCResult:
    speed_mps: float
    steering_rad: float
    ref_path: np.ndarray
    predicted_path: np.ndarray | None
    solved: bool


class KinematicMPCTracker:
    """ROS-free MPC tracker migrated from lab-8."""

    def __init__(self, waypoint_csv: str | Path, config: MPCConfig):
        if cvxpy is None or block_diag is None or csc_matrix is None:
            raise RuntimeError("MPC mode requires cvxpy and scipy. Install project dependencies before using --mode mpc.")
        self.config = config
        self.waypoints = self._load_waypoints(waypoint_csv)
        self.last_accel = None
        self.last_delta = None
        self._init_problem()

    @staticmethod
    def _load_waypoints(path: str | Path):
        data = np.loadtxt(str(path), delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 4:
            raise ValueError(f"MPC waypoint file needs columns x,y,yaw,v; got shape {data.shape}")
        return data[:, 0].astype(float), data[:, 1].astype(float), data[:, 2].astype(float), data[:, 3].astype(float)

    def compute(self, state: VehicleState) -> MPCResult:
        cx, cy, cyaw, cv = self.waypoints
        ref_path = calc_interpolated_ref_trajectory(
            state.x,
            state.y,
            cx,
            cy,
            cv,
            cyaw,
            self.config.dt_s,
            self.config.horizon,
        )
        ref_path = self._align_ref_yaw(ref_path, state.yaw)
        x0 = np.array([state.x, state.y, state.speed, state.yaw], dtype=float)
        self.last_accel, self.last_delta, ox, oy, oyaw, ov, predicted = self._linear_mpc_control(
            ref_path,
            x0,
            self.last_accel,
            self.last_delta,
        )
        if self.last_accel is None or self.last_delta is None:
            return MPCResult(0.0, 0.0, ref_path, predicted, solved=False)

        steer = float(self.config.steering_sign * self.last_delta[0])
        v_mpc = state.speed + self.last_accel[0] * self.config.dt_s
        v_ref = float(ref_path[2, min(1, ref_path.shape[1] - 1)])
        gain = float(np.clip(self.config.speed_track_gain, 0.0, 1.0))
        speed = float(np.clip((1.0 - gain) * v_mpc + gain * v_ref, self.config.min_speed_mps, self.config.max_speed_mps))
        predicted_path = np.column_stack([ox, oy]) if ox is not None and oy is not None else None
        return MPCResult(speed, steer, ref_path, predicted_path, solved=True)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _align_ref_yaw(self, ref_path: np.ndarray, yaw0: float) -> np.ndarray:
        ref_path = ref_path.copy()
        ref_path[3, 0] = yaw0 + self._normalize_angle(ref_path[3, 0] - yaw0)
        for i in range(1, ref_path.shape[1]):
            ref_path[3, i] = ref_path[3, i - 1] + self._normalize_angle(ref_path[3, i] - ref_path[3, i - 1])
        return ref_path

    def _init_problem(self) -> None:
        cfg = self.config
        self.xk = cvxpy.Variable((cfg.nx, cfg.horizon + 1))
        self.uk = cvxpy.Variable((cfg.nu, cfg.horizon))
        self.x0k = cvxpy.Parameter((cfg.nx,))
        self.ref_traj_k = cvxpy.Parameter((cfg.nx, cfg.horizon + 1))

        r_block = block_diag(tuple([cfg.input_cost] * cfg.horizon))
        rd_block = block_diag(tuple([cfg.input_rate_cost] * (cfg.horizon - 1)))
        q_block = block_diag(tuple([cfg.state_cost] * cfg.horizon + [cfg.final_state_cost]))

        objective = cvxpy.quad_form(cvxpy.vec(self.uk, order="F"), r_block)
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k, order="F"), q_block)
        objective += cvxpy.quad_form(cvxpy.vec(self.uk[:, 1:] - self.uk[:, :-1], order="F"), rd_block)

        path_predict = np.zeros((cfg.nx, cfg.horizon + 1))
        a_block, b_block, c_block = self._linearized_blocks(path_predict)
        self.Annz_k, self.Ak_ = self._sparse_parameter(a_block)
        self.Bnnz_k, self.Bk_ = self._sparse_parameter(b_block)
        self.Ck_ = cvxpy.Parameter(c_block.shape)
        self.Ck_.value = c_block

        constraints = [
            cvxpy.vec(self.xk[:, 1:], order="F")
            == self.Ak_ @ cvxpy.vec(self.xk[:, :-1], order="F")
            + self.Bk_ @ cvxpy.vec(self.uk, order="F")
            + self.Ck_,
            self.xk[:, 0] == self.x0k,
            self.xk[2, :] <= cfg.max_speed_mps,
            self.xk[2, :] >= cfg.min_speed_mps,
            cvxpy.abs(self.uk[0, :]) <= cfg.max_accel_mps2,
            cvxpy.abs(self.uk[1, :]) <= cfg.max_steer_rad,
            cvxpy.abs(self.uk[1, 1:] - self.uk[1, :-1]) <= cfg.max_dsteer_radps * cfg.dt_s,
        ]
        self.problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    @staticmethod
    def _sparse_parameter(matrix):
        m, n = matrix.shape
        param = cvxpy.Parameter(matrix.nnz)
        data = np.ones(param.size)
        rows = matrix.row * n + matrix.col
        cols = np.arange(param.size)
        indexer = csc_matrix((data, (rows, cols)), shape=(m * n, param.size))
        param.value = matrix.data
        return param, cvxpy.reshape(indexer @ param, (m, n), order="C")

    def _linearized_blocks(self, path_predict):
        a_blocks = []
        b_blocks = []
        c_values = []
        for t in range(self.config.horizon):
            a, b, c = self._model_matrix(path_predict[2, t], path_predict[3, t], 0.0)
            a_blocks.append(a)
            b_blocks.append(b)
            c_values.extend(c)
        return block_diag(tuple(a_blocks)), block_diag(tuple(b_blocks)), np.array(c_values)

    def _model_matrix(self, v: float, yaw: float, delta: float):
        cfg = self.config
        a = np.eye(cfg.nx)
        a[0, 2] = cfg.dt_s * math.cos(yaw)
        a[0, 3] = -cfg.dt_s * v * math.sin(yaw)
        a[1, 2] = cfg.dt_s * math.sin(yaw)
        a[1, 3] = cfg.dt_s * v * math.cos(yaw)
        a[3, 2] = cfg.dt_s * math.tan(delta) / cfg.wheelbase_m

        b = np.zeros((cfg.nx, cfg.nu))
        b[2, 0] = cfg.dt_s
        b[3, 1] = cfg.dt_s * v / (cfg.wheelbase_m * math.cos(delta) ** 2)

        c = np.zeros(cfg.nx)
        c[0] = cfg.dt_s * v * math.sin(yaw) * yaw
        c[1] = -cfg.dt_s * v * math.cos(yaw) * yaw
        c[3] = -cfg.dt_s * v * delta / (cfg.wheelbase_m * math.cos(delta) ** 2)
        return a, b, c

    def _predict_motion(self, x0, accel, delta, xref):
        predicted = xref * 0.0
        predicted[:, 0] = x0
        x, y, v, yaw = x0
        for i, (a, d) in enumerate(zip(accel, delta), start=1):
            d = float(np.clip(d, self.config.min_steer_rad, self.config.max_steer_rad))
            x += v * math.cos(yaw) * self.config.dt_s
            y += v * math.sin(yaw) * self.config.dt_s
            yaw = self._normalize_angle(yaw + (v / self.config.wheelbase_m) * math.tan(d) * self.config.dt_s)
            v = float(np.clip(v + a * self.config.dt_s, self.config.min_speed_mps, self.config.max_speed_mps))
            predicted[:, i] = [x, y, v, yaw]
        return predicted

    def _solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0
        a_block, b_block, c_block = self._linearized_blocks(path_predict)
        self.Annz_k.value = a_block.data
        self.Bnnz_k.value = b_block.data
        self.Ck_.value = c_block
        self.ref_traj_k.value = ref_traj
        self.problem.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if self.problem.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
            return None, None, None, None, None, None
        return (
            np.array(self.uk.value[0, :]).flatten(),
            np.array(self.uk.value[1, :]).flatten(),
            np.array(self.xk.value[0, :]).flatten(),
            np.array(self.xk.value[1, :]).flatten(),
            np.array(self.xk.value[3, :]).flatten(),
            np.array(self.xk.value[2, :]).flatten(),
        )

    def _linear_mpc_control(self, ref_path, x0, accel, delta):
        if accel is None or delta is None:
            accel = [0.0] * self.config.horizon
            delta = [0.0] * self.config.horizon
        path_predict = self._predict_motion(x0, accel, delta, ref_path)
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self._solve(ref_path, path_predict, x0)
        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict
