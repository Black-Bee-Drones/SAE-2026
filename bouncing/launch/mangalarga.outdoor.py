from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bouncing',
            executable='mangalarga',
            name='mangalarga',
            output='screen',
        )
    ])
