from setuptools import find_packages, setup
import os
from glob import glob

package_name = "mobile_actuator"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "launch", "launch.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "mobile_actuator_node.py = mobile_actuator.mobile_actuator_node:main",
        ],
    },
)