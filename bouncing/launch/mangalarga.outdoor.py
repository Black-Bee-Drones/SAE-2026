from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bouncing',
            executable='mangalarga',
            name='mangalarga',
            output='screen',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('bouncing'),
                    'config',
                    'default.yaml'
                ]),
            ]
        )
    ])
