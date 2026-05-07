"""SAE 2026 Hook simulation launch.

Spawns the arena world with optional drone yaw and sphere position via
launch args. The canonical SDF is read, its <pose> tags for `iris` and
`sphere_marker` are rewritten, and the result is written to a temp file
that is passed to `sitl_gazebo.launch.py`.

Args:
    drone_yaw_deg (float, default 0)   : iris yaw at spawn, degrees
    drone_x, drone_y (float, default 0): iris XY at spawn, meters
    sphere_x (float, default 2.0)      : sphere X on hose, meters
    sphere_y (float, default 1.25)     : sphere Y on hose, meters
    fcu_url (str)                      : MAVLink URL forwarded to sitl_gazebo

Example:
    ros2 launch hook sae_hook.launch.py \
        drone_yaw_deg:=45 sphere_x:=-2.5 sphere_y:=1.25
"""
import os
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DRONE_SPAWN_Z = 0.195
SPHERE_Z = 1.7


def _render_world(template_path: str, drone_pose: str, sphere_pose: str) -> str:
    """Rewrite iris and sphere_marker poses in the SDF; return temp path."""
    tree = ET.parse(template_path)
    root = tree.getroot()

    # sphere_marker is a top-level <model>
    for model in root.iter("model"):
        if model.get("name") == "sphere_marker":
            pose = model.find("pose")
            if pose is not None:
                pose.text = sphere_pose
            break

    # iris is included via <include><name>iris</name>...</include>
    for inc in root.iter("include"):
        name = inc.find("name")
        if name is not None and name.text == "iris":
            pose = inc.find("pose")
            if pose is not None:
                pose.set("degrees", "true")
                pose.text = drone_pose
            break

    fd, out_path = tempfile.mkstemp(prefix="sae_hook_arena_", suffix=".sdf")
    os.close(fd)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def _launch_setup(context: LaunchContext):
    hook_share = get_package_share_directory("hook")
    nectar_share = get_package_share_directory("nectar")

    template = os.path.join(hook_share, "simulation", "worlds", "sae_hook_arena.sdf")

    drone_x = float(LaunchConfiguration("drone_x").perform(context))
    drone_y = float(LaunchConfiguration("drone_y").perform(context))
    drone_yaw = float(LaunchConfiguration("drone_yaw_deg").perform(context))
    sphere_x = float(LaunchConfiguration("sphere_x").perform(context))
    sphere_y = float(LaunchConfiguration("sphere_y").perform(context))

    drone_pose = f"{drone_x} {drone_y} {DRONE_SPAWN_Z} 0 0 {drone_yaw}"
    sphere_pose = f"{sphere_x} {sphere_y} {SPHERE_Z} 0 0 0"

    world_path = _render_world(template, drone_pose, sphere_pose)
    print(f"[sae_hook] rendered world -> {world_path}")
    print(f"[sae_hook] iris pose (deg): {drone_pose}")
    print(f"[sae_hook] sphere pose:    {sphere_pose}")

    sitl_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nectar_share, "launch", "sitl_gazebo.launch.py")
        ),
        launch_arguments={
            "world": world_path,
            "fcu_url": LaunchConfiguration("fcu_url"),
        }.items(),
    )

    down_camera_republish = Node(
        package="image_transport",
        executable="republish",
        name="down_camera_republish",
        arguments=["raw", "compressed"],
        remappings=[
            ("in", "/down_camera"),
            ("out/compressed", "/down_camera/compressed"),
        ],
        output="log",
    )

    return [sitl_gazebo, down_camera_republish]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("fcu_url", default_value="tcp://127.0.0.1:5760"),
        DeclareLaunchArgument("drone_x", default_value="0.0"),
        DeclareLaunchArgument("drone_y", default_value="0.0"),
        DeclareLaunchArgument("drone_yaw_deg", default_value="0.0"),
        DeclareLaunchArgument("sphere_x", default_value="2.0"),
        DeclareLaunchArgument("sphere_y", default_value="1.25"),
        OpaqueFunction(function=_launch_setup),
    ])
