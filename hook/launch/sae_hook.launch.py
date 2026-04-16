import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    hook_share = get_package_share_directory("hook")
    nectar_share = get_package_share_directory("nectar")

    world_path = os.path.join(
        hook_share, "simulation", "worlds", "sae_hook_arena.sdf"
    )

    fcu_url_arg = DeclareLaunchArgument(
        "fcu_url",
        default_value="tcp://127.0.0.1:5760",
        description="MAVLink connection URL to ArduPilot SITL",
    )

    sitl_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nectar_share, "launch", "sitl_gazebo.launch.py")
        ),
        launch_arguments={
            "world": world_path,
            "fcu_url": LaunchConfiguration("fcu_url"),
        }.items(),
    )

    return LaunchDescription([fcu_url_arg, sitl_gazebo])
