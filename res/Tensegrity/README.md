# Importing Tensegrity into Isaac Sim / Isaac Lab

## Changes to URDF for IsaacSim (from initial Gazebo specification)

### 1. Remove Gazebo spezific files

Needed Files:

- URDF/Xacro
        - urdf/threedof_manipulator.urdf.xacro (main)
        - urdf/colors.urdf.xacro (included by main)
- Meshes referenced by the robot
        - (meshes/Rahmen.stl)
        - meshes/Oberarm.stl
        - meshes/Unterarm_Simulation.stl
        - meshes/Endeffektor.stl

### 2. Expand Xacro -> URDF

```bash
ros2 run xacro xacro --inorder -o threedof_manipulator.urdf threedof_manipulator.urdf.xacro
```

ROS2 needed.   
For minimal xacro without ros2:  
```bash
# install necessary packages
python3 -m pip install -U xacro rospkg catkin_pkg
export ROS_PACKAGE_PATH="$(pwd):$ROS_PACKAGE_PATH"   # run from Tensegrity/

# call xacro without relying on entry-point scripts:  
python3 - <<'PY'
import sys, xacro
sys.argv = ["xacro", "--inorder",
            "-o", "urdf/threedof_manipulator.urdf",
            "urdf/threedof_manipulator.urdf.xacro"]
xacro.main()
PY

# uninstall packages
python3 -m pip uninstall -y xacro rospkg catkin_pkg
```

### 3. Replace absolute paths to meshes in URDF references

```bash
python3 - <<'PY'
import pathlib, re

old_path = "package://robot_description/"
new_path = "/home/robot/studentische-arbeiten/res/Tensegrity/"

urdf = pathlib.Path(new_path + "/urdf/threedof_manipulator.urdf")
txt = urdf.read_text()
txt = txt.replace(old_path, new_path)
urdf.write_text(txt)
print("Rewrote mesh URIs to:", new_path)
PY
```

### 4. Remove Gazebo specific helper links/joints 

ForceX/IMUX links/joints are scaffolds for Gazebo-plugins, which can not be used within IsaacSim/Lab. Equivalent data can be read out in IsaacSim directly. For Sensor simulation, readd IMUs within IsaacSim USD file.

## Import Specifics

> URDF-Importer: File > Import > Select File  

### URDF Importer Settings (Recommended)
- **Merge Fixed Joints:** On  
- **Decompose Convex Meshes:** On  
- **Fix Base:** On
- **Self-Collision:** Off (initially) 
