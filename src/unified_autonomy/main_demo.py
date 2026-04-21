from __future__ import annotations

import argparse

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from unified_autonomy.adapters.mpc_adapter import MPCAdapter
from unified_autonomy.adapters.reactive_adapter import ReactiveFollowGapAdapter
from unified_autonomy.adapters.rl_adapter import RLAdapter
from unified_autonomy.adapters.rrt_adapter import RRTReactiveAdapter
from unified_autonomy.config import DemoConfig, load_config
from unified_autonomy.dashboard_state import DashboardStateStore, VALID_MODES
from unified_autonomy.interfaces import ControlCommand, PerceptionFrame, VehicleState, state_from_odom
from unified_autonomy.metrics import MetricsLogger
from unified_autonomy.visualization import VisualizationHooks, path_from_xy


class MainDemo(Node):
    def __init__(self, config: DemoConfig, mode_name: str):
        super().__init__("unified_autonomy_demo")
        self.config = config
        self.mode_name = mode_name
        self.requested_mode_name = mode_name
        self.state = VehicleState()
        self.perception = PerceptionFrame()
        self.last_command = ControlCommand(speed=0.0, steering_angle=0.0, mode=mode_name)
        self.last_dashboard_error = ""

        topics = config.topics
        self.drive_pub = self.create_publisher(AckermannDriveStamped, topics.get("drive", "/drive"), 10)
        self.create_subscription(LaserScan, topics.get("scan", "/scan"), self.scan_callback, 10)
        self.create_subscription(Odometry, topics.get("odom", "/ego_racecar/odom"), self.odom_callback, 10)

        self.frame_id = config.raw.get("project", {}).get("map_frame", "map")
        self.viz = VisualizationHooks(self, topics, self.frame_id)
        output_dir = config.resolve(config.metrics.get("output_dir", "logs"))
        self.metrics = MetricsLogger(
            output_dir=output_dir,
            run_tag=str(config.metrics.get("run_tag", "demo")),
            collision_distance_m=float(config.safety.get("collision_distance_m", 0.18)),
        )
        dashboard_cfg = config.raw.get("dashboard", {})
        self.dashboard_enabled = bool(dashboard_cfg.get("enabled", True))
        dashboard_state_file = config.resolve(str(dashboard_cfg.get("state_file", "dashboard/state.json")))
        self.dashboard_store = DashboardStateStore(dashboard_state_file) if self.dashboard_enabled else None
        self.mode = self.make_mode(mode_name)
        self.publish_dashboard_state(status="running")
        rate_hz = float(config.runtime.get("control_rate_hz", 20.0))
        self.timer = self.create_timer(1.0 / rate_hz, self.control_tick)
        self.get_logger().info(f"Unified autonomy demo running in mode={mode_name}")

    def make_mode(self, mode_name: str):
        mode_cfg = self.config.mode_config(mode_name)
        assets = self.config.assets
        if mode_name == "reactive":
            return ReactiveFollowGapAdapter(mode_cfg, self.config.safety)
        if mode_name == "rrt":
            return RRTReactiveAdapter(mode_cfg, self.config.safety)
        if mode_name == "mpc":
            waypoint_csv = assets.get("waypoint_csv")
            waypoint_path = str(self.config.resolve(waypoint_csv)) if waypoint_csv else None
            return MPCAdapter(mode_cfg, self.config.safety, waypoint_csv=waypoint_path)
        if mode_name == "rl":
            policy_path = assets.get("rl_policy_path")
            return RLAdapter(mode_cfg, self.config.safety, policy_path=policy_path)
        raise ValueError(f"Unknown mode: {mode_name}")

    def scan_callback(self, msg: LaserScan) -> None:
        self.perception.scan = msg
        self.perception.obstacles_xy = self.scan_to_points(msg)

    def odom_callback(self, msg: Odometry) -> None:
        self.state = state_from_odom(msg)

    def control_tick(self) -> None:
        self.apply_dashboard_mode_request()
        try:
            output = self.mode.update(self.state, self.perception)
        except NotImplementedError as exc:
            self.get_logger().warn(str(exc))
            output = self.safe_stop("mode_not_implemented")

        command = self.apply_safety_limits(output.command)
        self.publish_drive(command)
        self.metrics.update(self.state, self.perception, command)
        planned_path = output.planned_path
        if isinstance(planned_path, np.ndarray):
            planned_path = path_from_xy(planned_path, self.frame_id, self.get_clock().now().to_msg())
        self.viz.publish_path(planned_path)
        self.viz.publish_goal(output.goal_xy)
        self.viz.publish_obstacles(output.obstacle_points if output.obstacle_points is not None else self.perception.obstacles_xy)
        self.viz.publish_status(self.state, command.mode, text=f"v={self.state.speed:.2f}")
        self.last_command = command
        self.publish_dashboard_state(status="running")

    def apply_dashboard_mode_request(self) -> None:
        if self.dashboard_store is None:
            return
        state = self.dashboard_store.read()
        requested = str(state.get("requested_mode", self.mode_name)).lower().strip()
        self.requested_mode_name = requested
        if requested == self.mode_name:
            return
        if requested not in VALID_MODES:
            self.last_dashboard_error = f"Invalid requested mode: {requested}"
            return
        try:
            next_mode = self.make_mode(requested)
        except Exception as exc:
            self.last_dashboard_error = f"Could not switch to {requested.upper()}: {type(exc).__name__}: {exc}"
            self.get_logger().warn(self.last_dashboard_error)
            self.publish_dashboard_state(status="mode_switch_failed")
            return
        self.mode = next_mode
        self.mode_name = requested
        self.last_dashboard_error = ""
        self.get_logger().info(f"Dashboard switched mode to {requested}")
        self.publish_dashboard_state(status="running")

    def safe_stop(self, reason: str):
        from unified_autonomy.interfaces import ModeOutput

        return ModeOutput(
            command=ControlCommand(
                speed=0.0,
                steering_angle=0.0,
                mode=self.mode_name,
                debug={"safe_stop": True, "reason": reason},
            )
        )

    def apply_safety_limits(self, command: ControlCommand) -> ControlCommand:
        max_speed = float(self.config.safety.get("max_speed_mps", 4.0))
        max_steer = float(self.config.safety.get("max_steer_rad", 0.42))
        command.speed = float(np.clip(command.speed, 0.0, max_speed))
        command.steering_angle = float(np.clip(command.steering_angle, -max_steer, max_steer))
        return command

    def publish_drive(self, command: ControlCommand) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.config.raw.get("project", {}).get("base_frame", "ego_racecar/base_link")
        msg.drive.speed = command.speed
        msg.drive.steering_angle = command.steering_angle
        self.drive_pub.publish(msg)

    def publish_dashboard_state(self, status: str) -> None:
        if self.dashboard_store is None:
            return
        snap = self.metrics.snapshot(status=status)
        self.dashboard_store.update_runtime(
            active_mode=self.mode_name,
            requested_mode=self.requested_mode_name,
            status=status,
            last_error=self.last_dashboard_error,
            metrics=snap.__dict__,
            command={
                "speed": float(self.last_command.speed),
                "steering_angle": float(self.last_command.steering_angle),
            },
            vehicle={
                "x": float(self.state.x),
                "y": float(self.state.y),
                "yaw": float(self.state.yaw),
                "speed": float(self.state.speed),
            },
        )

    @staticmethod
    def scan_to_points(msg: LaserScan) -> np.ndarray:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        valid = np.isfinite(ranges)
        if not np.any(valid):
            return np.empty((0, 2))
        angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        return np.column_stack([x, y])

    def destroy_node(self) -> bool:
        self.publish_dashboard_state(status="stopped")
        self.metrics.close(status="stopped")
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--mode", choices=["reactive", "rrt", "mpc", "rl"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mode = args.mode or str(config.runtime.get("default_mode", "reactive"))
    rclpy.init()
    node = MainDemo(config, mode)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
