from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from unified_autonomy.interfaces import VehicleState


class VisualizationHooks:
    """Publish standard ROS messages that Foxglove Studio can visualize directly."""

    def __init__(self, node: Node, topics: dict[str, str], frame_id: str):
        self.node = node
        self.frame_id = frame_id
        self.path_pub = node.create_publisher(Path, topics.get("planned_path", "/unified/planned_path"), 10)
        self.goal_pub = node.create_publisher(Marker, topics.get("goal_marker", "/unified/goal"), 10)
        self.obstacle_pub = node.create_publisher(MarkerArray, topics.get("obstacle_markers", "/unified/obstacles"), 10)
        self.status_pub = node.create_publisher(Marker, topics.get("status_marker", "/unified/status"), 10)

    def publish_path(self, path: Path | None) -> None:
        if path is not None:
            self.path_pub.publish(path)

    def publish_goal(self, goal_xy: tuple[float, float] | None) -> None:
        if goal_xy is None:
            return
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "unified_goal"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal_xy[0])
        marker.pose.position.y = float(goal_xy[1])
        marker.pose.position.z = 0.1
        marker.scale.x = marker.scale.y = marker.scale.z = 0.25
        marker.color = ColorRGBA(r=0.1, g=0.8, b=0.2, a=0.9)
        self.goal_pub.publish(marker)

    def publish_obstacles(self, points_xy: np.ndarray | None) -> None:
        if points_xy is None or points_xy.size == 0:
            return
        markers = MarkerArray()
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "unified_obstacles"
        marker.id = 1
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color = ColorRGBA(r=0.9, g=0.1, b=0.1, a=0.8)
        for x, y in points_xy[:500]:
            marker.points.append(Point(x=float(x), y=float(y), z=0.05))
        markers.markers.append(marker)
        self.obstacle_pub.publish(markers)

    def publish_status(self, state: VehicleState, mode: str, text: str = "") -> None:
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "unified_status"
        marker.id = 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = state.x
        marker.pose.position.y = state.y
        marker.pose.position.z = 0.8
        marker.scale.z = 0.22
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
        marker.text = f"mode={mode} {text}".strip()
        self.status_pub.publish(marker)


def path_from_xy(points_xy: np.ndarray, frame_id: str, stamp) -> Path:
    path = Path()
    path.header.frame_id = frame_id
    if stamp is not None:
        path.header.stamp = stamp
    for x, y in points_xy:
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    return path
