"""Build a Gazebo (Harmonic) world from the Microban MJCF.

Every MJCF body becomes its own pose-driven Gazebo model, twice: `live_<body>`
(the real robot) and `ghost_<body>` (a translucent reference/target pose). The
viewer computes forward kinematics with MuJoCo from the same MJCF the walking
policy was trained on and moves all models at once through the
/world/mero/set_pose_vector service, so joint zero and direction conventions are
exactly the simulation's.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MJCF = ROOT / "model" / "mjcf" / "robot.xml"
WORLD_NAME = "mero"

GHOST_RGBA = (0.1, 0.85, 1.0, 0.6)

# Start the camera behind-right of the robot (robot faces +x, its left is +y), looking the same
# way it does: the robot's left leg is on the left of the screen, like standing behind the real one.
CAMERA_POSE = "-0.55 -0.45 0.42 0 0.35 0.686"


def _floats(s: str | None, default):
    return tuple(float(v) for v in s.split()) if s else default


def _quat_to_rpy(q):
    """SDF fixed-axis roll/pitch/yaw (R = Rz(yaw) Ry(pitch) Rx(roll)), gimbal-lock safe."""
    import math
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    if math.hypot(r00, r10) > 1e-6:
        roll = math.atan2(r21, r22)
        yaw = math.atan2(r10, r00)
    else:   # pitch = +-90 deg: put everything into roll
        roll = math.atan2(-r12, r11)
        yaw = 0.0
    return roll, pitch, yaw


def parse_bodies(mjcf: Path = MJCF) -> dict[str, list[dict]]:
    """body name -> list of visual meshes {file, pos, quat, rgba}."""
    root = ET.parse(mjcf).getroot()
    meshdir = mjcf.parent / root.find("compiler").get("meshdir", ".")
    meshes = {}
    for m in root.iter("mesh"):
        f = m.get("file")
        meshes[m.get("name") or Path(f).stem] = (meshdir / f).resolve()
    mats = {m.get("name"): _floats(m.get("rgba"), (0.6, 0.6, 0.6, 1)) for m in root.iter("material")}
    bodies: dict[str, list[dict]] = {}
    for b in root.iter("body"):
        vis = []
        for g in b.findall("geom"):
            if g.get("class") != "visual" or g.get("mesh") is None:
                continue
            vis.append({
                "file": meshes[g.get("mesh")],
                "pos": _floats(g.get("pos"), (0.0, 0.0, 0.0)),
                "quat": _floats(g.get("quat"), (1.0, 0.0, 0.0, 0.0)),
                "rgba": mats.get(g.get("material"), (0.6, 0.6, 0.6, 1.0)),
            })
        bodies[b.get("name")] = vis
    return bodies


def _model(name: str, visuals: list[dict], ghost: bool, z_hidden: float) -> str:
    out = [f'    <model name="{name}">', f"      <pose>0 0 {z_hidden} 0 0 0</pose>",
           '      <link name="link">', "        <gravity>false</gravity>",
           "        <inertial><mass>0.01</mass><inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz>"
           "<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>"]
    for k, v in enumerate(visuals):
        r, p, y = _quat_to_rpy(v["quat"])
        rgba = GHOST_RGBA if ghost else v["rgba"]
        col = " ".join(f"{c:.3f}" for c in rgba)
        out += [f'        <visual name="v{k}">',
                f"          <pose>{v['pos'][0]} {v['pos'][1]} {v['pos'][2]} {r:.6f} {p:.6f} {y:.6f}</pose>",
                f"          <geometry><mesh><uri>file://{v['file'].as_posix()}</uri></mesh></geometry>",
                f"          <material><ambient>{col}</ambient><diffuse>{col}</diffuse><specular>0.1 0.1 0.1 1</specular></material>"]
        if ghost:
            out.append(f"          <transparency>{1 - rgba[3]:.2f}</transparency>")
        out.append("        </visual>")
    out += ["      </link>", "    </model>"]
    return "\n".join(out)


GUI = """  <gui fullscreen="0">
    <plugin filename="MinimalScene" name="3D View">
      <gz-gui><title>3D View</title><property type="bool" key="showTitleBar">false</property>
        <property type="string" key="state">docked</property></gz-gui>
      <engine>ogre2</engine><scene>scene</scene>
      <ambient_light>0.5 0.5 0.5</ambient_light><background_color>0.75 0.8 0.85</background_color>
      <camera_pose>{CAMERA_POSE}</camera_pose>
      <camera_clip><near>0.01</near><far>50</far></camera_clip>
    </plugin>
    <plugin filename="GzSceneManager" name="Scene Manager"><gz-gui><property key="state" type="string">floating</property>
      <property key="width" type="double">5</property><property key="height" type="double">5</property>
      <property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
    <plugin filename="InteractiveViewControl" name="Interactive view control"><gz-gui><property key="state" type="string">floating</property>
      <property key="width" type="double">5</property><property key="height" type="double">5</property>
      <property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
    <plugin filename="CameraTracking" name="Camera Tracking"><gz-gui><property key="state" type="string">floating</property>
      <property key="width" type="double">5</property><property key="height" type="double">5</property>
      <property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
    <plugin filename="MarkerManager" name="Marker manager"><gz-gui><property key="state" type="string">floating</property>
      <property key="width" type="double">5</property><property key="height" type="double">5</property>
      <property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
    <plugin filename="SelectEntities" name="Select Entities"><gz-gui><property key="state" type="string">floating</property>
      <property key="width" type="double">5</property><property key="height" type="double">5</property>
      <property key="showTitleBar" type="bool">false</property></gz-gui></plugin>
    <plugin filename="WorldControl" name="World control"><gz-gui><title>World control</title>
      <property type="bool" key="showTitleBar">false</property><property type="string" key="state">floating</property>
      <anchors target="3D View"><line own="left" target="left"/><line own="bottom" target="bottom"/></anchors></gz-gui>
      <play_pause>true</play_pause><step>false</step><start_paused>false</start_paused></plugin>
  </gui>
"""


def build_world(path: Path, mjcf: Path = MJCF, engine: str = "ogre2") -> list[str]:
    bodies = parse_bodies(mjcf)
    models = []
    for name, vis in bodies.items():
        models.append(_model(f"live_{name}", vis, False, -5))
        models.append(_model(f"ghost_{name}", vis, True, -10))
    sdf = f"""<?xml version="1.0"?>
<!-- generated by mero_walk/viewer/gz_world.py from {mjcf.name} -->
<sdf version="1.9">
  <world name="{WORLD_NAME}">
    <physics name="4ms" type="dart"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <gravity>0 0 0</gravity>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
{GUI.replace("<engine>ogre2</engine>", f"<engine>{engine}</engine>").replace("{CAMERA_POSE}", CAMERA_POSE)}
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.75 0.8 0.85 1</background><grid>false</grid></scene>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 3 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.4 0.3 -0.9</direction></light>
    <model name="ground"><static>true</static><link name="link"><visual name="v">
      <geometry><plane><normal>0 0 1</normal><size>4 4</size></plane></geometry>
      <material><ambient>0.55 0.6 0.65 1</ambient><diffuse>0.55 0.6 0.65 1</diffuse></material></visual></link></model>
{chr(10).join(models)}
  </world>
</sdf>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sdf, encoding="utf-8")
    return list(bodies)


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "viewer" / "mero_world.sdf"
    names = build_world(out)
    print(f"wrote {out} ({len(names)} bodies x2)")
