import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('rrt_pure_mix')
    default_params = os.path.join(pkg_share, 'config', 'rrt_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the RRT params YAML file'
    )

    rrt_node = Node(
        package='rrt_pure_mix',
        executable='rrt_node',
        name='rrt_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        rrt_node,
    ])
