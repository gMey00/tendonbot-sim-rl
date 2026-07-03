# Robot Hardware & Datasheets

[← Literature Overview](README.md) · [← Project README](../../README.md)

Manufacturer datasheets, technical specifications, and user guides for the
commercial robot arms used as comparison baselines in the master thesis
(UR5e, UR10, Kinova Gen3) alongside the FAPS tensegrity manipulator.
These primary sources back the figures in
[doc/robot_gripper_comparison.md](../robot_gripper_comparison.md).

> **Grippers:** the Robotiq adaptive-gripper datasheet is catalogued under
> [simulation.md → Motion Planning](simulation.md) as
> [`Robotiq2F140Datasheet`](sources/simulation/Documentation/Robotiq2F140Datasheet.pdf).

---

## Collaborative & Research Robot Arms

### Datasheets

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| [UniversalRobotsUR5e](sources/robot_hardware/Datasheets/UniversalRobotsUR5e.pdf) | Universal Robots A/S | UR5e Technical Specification | 🟡 | 🟢 | 🟡 | 6-DOF e-Series cobot, 5 kg payload, 850 mm reach, ±0.03 mm repeatability, 20.6 kg, Ø149 mm footprint. Primary comparison arm for the master thesis. [Link](https://www.universal-robots.com/products/ur5e/) |
| [UniversalRobotsUR10](sources/robot_hardware/Datasheets/UniversalRobotsUR10.pdf) | Universal Robots A/S | UR10 Technical Specifications (CB3-Series) | 🟡 | 🟢 | 🟡 | 6-DOF CB3 cobot, 10 kg payload, 1300 mm reach, ±0.1 mm repeatability, 28.9 kg, Ø190 mm footprint. Item no. 110110. [Link](https://www.universal-robots.com/media/50895/ur10_en.pdf) |
| [UniversalRobotsUR10e](sources/robot_hardware/Datasheets/UniversalRobotsUR10e.pdf) | Universal Robots A/S | UR10e Technical Specification | 🟡 | 🟡 | 🟡 | 6-DOF e-Series cobot, 12.5 kg payload, 1300 mm reach, ±0.05 mm repeatability, 33.5 kg. Reference for the e-Series platform. [Link](https://www.universal-robots.com/products/ur10e/) |
| [KinovaGen3Datasheet](sources/robot_hardware/Datasheets/KinovaGen3Datasheet.pdf) | Kinova Robotics | Gen3 Ultra Lightweight Robot — Technical Specifications | 🟡 | 🟢 | 🟡 | Two-page spec sheet: 7 DOF, 8.2 kg, 4 kg mid-range / 2 kg full-range payload, 902 mm reach, IP33, integrated torque/position/velocity/current sensors. [Link](https://www.kinovarobotics.com/product/gen3-robots) |
| [KinovaGen3Brochure](sources/robot_hardware/Datasheets/KinovaGen3Brochure.pdf) | Kinova Robotics | Gen3 Robot — Together in Robotics (Product Brochure & Specifications) | 🔴 | 🟡 | 🟡 | 2024 product brochure with 6-DOF (7.2 kg, 891 mm) and 7-DOF (8.2 kg, 902 mm) variants; ROS Noetic / ROS 2 Humble support. [Link](https://www.kinovarobotics.com/product/gen3-robots) |

### Manuals

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| [KinovaGen3UserGuide](sources/robot_hardware/Manuals/KinovaGen3UserGuide.pdf) | Kinova Robotics | Gen3 Ultra Lightweight Robot — User Guide | 🟢 | 🟢 | 🟢 | 218-page user guide (firmware 2.3.0): DH parameters, joint limits, homogeneous transforms, inertial parameters, link lengths, and actuator torque ratings (small 13/34 N·m, large 32–39/54 N·m nominal/peak). Authoritative source for Kinova kinematics and dynamics. [Link](https://www.kinovarobotics.com/resources) |
