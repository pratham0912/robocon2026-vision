from setuptools import find_packages, setup

package_name = "spearhead_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch",  ["launch/spearhead_detector.launch.py"]),
        ("share/" + package_name + "/config",  ["config/params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pratham0912",
    maintainer_email="pratham0912@github.com",
    description="ROS2 node: spearhead detection, 3-D localisation, and ID tracking for ABU Robocon 2026",
    license="MIT",
    entry_points={
        "console_scripts": [
            "spearhead_detector_node = spearhead_detector.spearhead_detector_node:main",
        ],
    },
)
