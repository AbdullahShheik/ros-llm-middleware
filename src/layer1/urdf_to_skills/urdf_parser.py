"""
urdf_parser.py
---------------
parses a URDF (XML) file and produces a structured, compact summary of the
robot's physical capabilities -- joint types, plugin references, sensors.
This summary (not the raw XML) is what gets handed to the LLM matcher,
since raw URDF is verbose/noisy and the LLM only needs the structural
signal, not full geometry/visual/collision detail.

Uses only the standard library (xml.etree.ElementTree) -- no extra
dependency needed to parse URDF, which is just XML.
"""

import xml.etree.ElementTree as ET
from collections import Counter


def parse_urdf(urdf_path):
    """
    Returns a structured dict:
    {
        "robot_name": str,
        "joints": [{"name": ..., "type": ...}, ...],
        "joint_type_counts": {"revolute": 3, "continuous": 2, ...},
        "links": [str, ...],
        "plugins": [{"filename": ..., "name": ...}, ...],
        "sensors": [{"type": ..., "name": ...}, ...],
    }
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    robot_name = root.attrib.get("name", "unknown_robot")

    joints = []
    for joint_el in root.findall("joint"):
        joints.append({
            "name": joint_el.attrib.get("name", "unnamed_joint"),
            "type": joint_el.attrib.get("type", "unknown"),
        })

    links = [link_el.attrib.get("name", "unnamed_link") for link_el in root.findall("link")]

    plugins = []
    for gazebo_el in root.findall("gazebo"):
        for plugin_el in gazebo_el.findall("plugin"):
            plugins.append({
                "name": plugin_el.attrib.get("name", ""),
                "filename": plugin_el.attrib.get("filename", ""),
            })

    sensors = []
    for gazebo_el in root.findall("gazebo"):
        for sensor_el in gazebo_el.findall("sensor"):
            sensors.append({
                "name": sensor_el.attrib.get("name", ""),
                "type": sensor_el.attrib.get("type", ""),
            })
    # Some URDFs put <sensor> directly under a <link>'s <gazebo> tag with a
    # reference attribute instead -- also scan globally as a fallback.
    for sensor_el in root.iter("sensor"):
        entry = {"name": sensor_el.attrib.get("name", ""), "type": sensor_el.attrib.get("type", "")}
        if entry not in sensors:
            sensors.append(entry)

    joint_type_counts = dict(Counter(j["type"] for j in joints))

    return {
        "robot_name": robot_name,
        "joints": joints,
        "joint_type_counts": joint_type_counts,
        "links": links,
        "plugins": plugins,
        "sensors": sensors,
    }


def summarize_for_llm(parsed):
    """
    Turns the structured parse result into a compact text block suitable
    for injecting into the LLM matching prompt.
    """
    lines = [f"Robot name (from URDF): {parsed['robot_name']}"]

    lines.append(f"\nJoints ({len(parsed['joints'])} total):")
    for jtype, count in parsed["joint_type_counts"].items():
        lines.append(f"  - {count}x {jtype}")
    joint_names = ", ".join(j["name"] for j in parsed["joints"])
    lines.append(f"  Joint names: {joint_names}")

    lines.append(f"\nLinks ({len(parsed['links'])} total): {', '.join(parsed['links'])}")

    if parsed["plugins"]:
        lines.append("\nGazebo plugins detected:")
        for p in parsed["plugins"]:
            lines.append(f"  - {p['name'] or '(unnamed)'}: {p['filename']}")
    else:
        lines.append("\nGazebo plugins detected: none")

    if parsed["sensors"]:
        lines.append("\nSensors detected:")
        for s in parsed["sensors"]:
            lines.append(f"  - {s['name'] or '(unnamed)'} (type: {s['type']})")
    else:
        lines.append("\nSensors detected: none")

    lines.append(
        "\nNote: URDF does not directly encode battery/power information -- "
        "judge power-monitoring capability from context (robot name, plugins, "
        "typical characteristics of this kind of platform), not from explicit URDF fields."
    )

    return "\n".join(lines)
