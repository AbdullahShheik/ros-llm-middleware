from setuptools import find_packages, setup

package_name = "world_model"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ROS-LLM Team",
    maintainer_email="manahil.jamilh@gmail.com",
    description="Shared world-context library (SDF/map/object-map -> environment context).",
    license="MIT",
    # Library only -- no nodes, so no console_scripts.
    entry_points={},
)
