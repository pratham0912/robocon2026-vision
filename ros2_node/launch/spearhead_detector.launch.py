from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    display_arg = DeclareLaunchArgument(
        "display",
        default_value="true",
        description="Open a live OpenCV display window (set false on headless systems)",
    )
    params_file = PathJoinSubstitution(
        [FindPackageShare("spearhead_detector"), "config", "params.yaml"]
    )
    node = Node(
        package="spearhead_detector",
        executable="spearhead_detector_node",
        name="spearhead_detector",
        output="screen",
        parameters=[
            params_file,
            {"display": LaunchConfiguration("display")},
        ],
    )
    return LaunchDescription([display_arg, node])
