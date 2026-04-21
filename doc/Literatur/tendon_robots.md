# Tendon-Driven & Cable Robots

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Cable-Driven Parallel Robots

### Books

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Pott2018](sources/tendon_robots/Books/Pott2018.pdf) | Pott, A. | *Cable-Driven Parallel Robots: Theory and Application* | 2018 | 🟡 | 🟢 | 🟢 | Comprehensive monograph covering workspace analysis, tension distribution, stiffness modelling, and trajectory planning for CDPRs. Provides the formal framework for kinetostatic duality between cable tensions and end-effector wrenches. [Springer](https://link.springer.com/book/10.1007/978-3-319-76138-1) |
| Bruckmann2013 | Bruckmann, T. & Pott, A. (Eds.) | *Cable-Driven Parallel Robots* (Mechanisms and Machine Science, Vol. 12) | 2013 | 🔴 | 🟡 | 🟢 | Edited volume collecting state-of-the-art CDPR contributions from leading research groups. Covers design, modelling, control, and applications — broad context for the tendon-driven approach. [Springer](https://link.springer.com/book/10.1007/978-3-642-31988-4) |
| [Lau2025CableCon](sources/tendon_robots/Books/Lau2025CableCon.epub) | Lau, D. et al. (Eds.) | *Cable-Driven Parallel Robots (CableCon 2025)* — Mechanisms and Machine Science | 2025 | 🔴 | 🟡 | 🟢 | Proceedings of the 7th International Conference on Cable-Driven Parallel Robots (CableCon 2025). Latest compilation of CDPR research on design, modelling, control, and applications. [Springer](https://link.springer.com/book/10.1007/978-3-031-84914-3) |
| [Niola2020IFToMM](sources/tendon_robots/Books/Niola2020IFToMM.pdf) | Niola, V. & Gasparetto, A. (Eds.) | *Advances in Italian Mechanism Science* (IFToMM ITALY 2020) | 2020 | 🔴 | 🟡 | 🟢 | Edited volume with contributions on cable-driven and parallel mechanisms from the IFToMM Italy 2020 conference. Broader mechanism-science context for CDPR research. [DOI](https://doi.org/10.1007/978-3-030-55807-9) |

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Borgstrom2009](sources/tendon_robots/Journal_Articles/Borgstrom2009.pdf) | Borgstrom, P. H. et al. | "Rapid Computation of Optimally Safe Tension Distributions for Parallel Cable-Driven Robots" | 2009 | 🟡 | 🟢 | 🟢 | Efficient algorithms for computing cable tension distributions satisfying positive-tension constraints while minimising total cable force. Applicable to the 5-tendon / 3-DOF mapping in the tensegrity manipulator. |
| [Oh2005](sources/tendon_robots/Journal_Articles/Oh2005.pdf) | Oh, S.-R. & Agrawal, S. K. | "Cable Suspended Planar Robots with Redundant Cables: Controllers with Positive Tensions" | 2005 | 🟡 | 🟢 | 🟢 | Theoretical basis for controlling cable-driven robots with redundant cables under positive-tension constraints. The pseudo-inverse tension distribution approach used in step-response tests follows this line of work. |
| [Dhakate2025](sources/tendon_robots/Journal_Articles/Dhakate2025.pdf) | Dhakate, R. et al. | "CaRoSaC: A Reinforcement Learning-Based Kinematic Control of Cable-Driven Parallel Robots by Addressing Cable Sag Through Simulation" | 2025 | 🟡 | 🟢 | 🟢 | RL-based kinematic control of CDPRs addressing cable sag in simulation. Directly relevant as a prior-work comparison for RL-controlled cable robots. [DOI](https://doi.org/10.1109/LRA.2025.3564225) |
| [Garrido2024](sources/tendon_robots/Journal_Articles/Garrido2024.pdf) | Garrido Campos, J. et al. | "Integrated Simulation and Control Environment for CDPRs" | 2024 | 🟡 | 🟡 | 🟢 | Integrated simulation and control environment for cable-driven parallel robots, relevant as a comparable simulation approach. [DOI](https://doi.org/10.1080/0951192X.2024.2389640) |
| [Feng2023](sources/tendon_robots/Journal_Articles/Feng2023.pdf) | Feng, J. et al. | "Tracking Control Strategy of Tendon-Driven Robotic Arm Under Adaptive Neural Network" | 2023 | 🟡 | 🟡 | 🟢 | Adaptive neural-network-based tracking controller for tendon-driven arms. Relevant comparator for learning-based tendon control. [DOI](https://doi.org/10.3389/fmech.2023.1281660) |

### Surveys

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Zhang2022survey](sources/tendon_robots/Surveys/Zhang2022survey.pdf) | Zhang, Z., Shao, Z., You, Z., Tang, X., Zi, B., Yang, G., Gosselin, C. & Caro, S. | "State-of-the-Art on Theories and Applications of Cable-Driven Parallel Robots" | 2022 | 🔴 | 🟢 | 🟢 | Comprehensive CDPM survey covering configuration design, force distribution, workspace analysis, stiffness, optimization, and control. |
| [Qian2018](sources/tendon_robots/Surveys/Qian2018.pdf) | Qian, S., Zi, B., Shang, W. W. & Xu, Q. S. | "A Review on Cable-driven Parallel Robots" | 2018 | 🔴 | 🟡 | 🟢 | CDPR history from Landsberger through NIST RoboCrane, with applications in 3D printing and construction. |
| [KinoKawamura2015](sources/tendon_robots/Surveys/KinoKawamura2015.pdf) | Kino, H. & Kawamura, S. | "Mechanism and Control of Parallel-Wire Driven System" | 2015 | 🔴 | 🟡 | 🟢 | Survey of mechanism design and control strategies for parallel-wire-driven systems. Provides historical and conceptual grounding for CDPR control. [DOI](https://doi.org/10.20965/jrm.2015.p0599) |

### Foundational Works

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Landsberger1984](sources/tendon_robots/Foundational_Works/Landsberger1984.pdf) | Landsberger, S. E. | *Design and Construction of a Cable-Controlled, Parallel Link Manipulator* | 1984 | 🔴 | 🟡 | 🟢 | First cable-driven parallel robot. Uses vertical compression members to maintain cable tension. Historical milestone. |
| Ming1994a | Ming, A. & Higuchi, T. | "Study on Multiple DOF Positioning Mechanism Using Wires (Part 1) — Concept, Design and Control" | 1994 | 🔴 | 🟢 | 🟢 | First m vs. n classification of cable robots: incompletely, completely, and redundantly restrained. |
| Ming1994b | Ming, A. & Higuchi, T. | "Study on Multiple DOF Positioning Mechanism Using Wires (Part 2) — Development of a Planar Completely Restrained Positioning Mechanism" | 1994 | 🔴 | 🟡 | 🟢 | Planar m = n + 1 prototype with force equilibrium equations for completely restrained positioning. |
| [Verhoeven2004](sources/tendon_robots/Foundational_Works/Verhoeven2004.pdf) | Verhoeven, R. | *Analysis of the Workspace of Tendon-based Stewart Platforms* | 2004 | 🟡 | 🟢 | 🟢 | Origin of the "structure matrix" terminology. IRPM/CRPM/RRPM classification and controllable workspace definition. |
| [Arai1991](sources/tendon_robots/Foundational_Works/Arai1991.pdf) | Arai, T. et al. | "A Study on Cable-Driven Parallel Mechanisms — Analysis and Design" | 1991 | 🔴 | 🟡 | 🟢 | Early foundational study on cable-driven parallel mechanism analysis and design. Historical CDPR milestone. [DOI](https://doi.org/10.1109/ROBOT.1991.131745) |

### Structure Matrix & Force Distribution

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Bruckmann2008a](sources/tendon_robots/Structure_Matrix_%26_Force_Distribution/Bruckmann2008a.pdf) | Bruckmann, T., Mikelsons, L., Brandt, T., Hiller, M. & Schramm, D. | "Wire Robots Part I: Kinematics, Analysis & Design" | 2008 | 🟡 | 🟢 | 🟢 | Explicit naming of the structure matrix; establishes J^T equivalence for cable robots. |
| [Bruckmann2008b](sources/tendon_robots/Structure_Matrix_%26_Force_Distribution/Bruckmann2008b.pdf) | Bruckmann, T., Mikelsons, L., Brandt, T., Hiller, M. & Schramm, D. | "Wire Robots Part II: Dynamics, Control & Application" | 2008 | 🟡 | 🟢 | 🟢 | A^T · t = w equation, dynamics, and force distribution methods (barycentric, LP, optimization). |
| [Pott2009](sources/tendon_robots/Structure_Matrix_%26_Force_Distribution/Pott2009.pdf) | Pott, A., Bruckmann, T. & Mikelsons, L. | "Closed-Form Force Distribution for Parallel Wire Robots" | 2009 | 🟡 | 🟢 | 🟢 | Non-iterative algorithm for force distribution with continuity guarantees. IPAnema prototype validation. |
| [Gosselin2011](sources/tendon_robots/Structure_Matrix_%26_Force_Distribution/Gosselin2011.pdf) | Gosselin, C. & Grenier, M. | "On the Determination of the Force Distribution in Overconstrained Cable-Driven Parallel Mechanisms" | 2011 | 🟡 | 🟢 | 🟢 | Wrench matrix terminology; p-norm optimization (4-norm) for force distribution via univariate polynomial. |
| Gosselin2008 | Gosselin, C. | "On the Determination of the Force Distribution in Overconstrained Cable-Driven Parallel Mechanisms" | 2008 | 🔴 | 🟡 | 🟢 | Conference version of Gosselin & Grenier (2011). Early wrench matrix force distribution formulation. |
| [EbertUphoff2004](sources/tendon_robots/Structure_Matrix_%26_Force_Distribution/EbertUphoff2004.pdf) | Ebert-Uphoff, I. & Voglewede, P. A. | "On the Connections Between Cable-Driven Robots, Parallel Manipulators and Grasping" | 2004 | 🔴 | 🟡 | 🟢 | Establishes the cable-driven ↔ grasping analogy; force-closure equivalence between cable tension and grasp contact. |
| [Voglewede2005](sources/tendon_robots/Structure_Matrix_%26_Force_Distribution/Voglewede2005.pdf) | Voglewede, P. A. & Ebert-Uphoff, I. | "Application of the Antipodal Grasp Theorem to Cable-Driven Robots" | 2005 | 🔴 | 🟡 | 🟢 | Ports the antipodal grasp theorem to cable robots for workspace and force analysis. |
| [Roberts1998](sources/tendon_robots/Structure_Matrix_%26_Force_Distribution/Roberts1998.pdf) | Roberts, R. G., Graham, T. & Lippitt, T. | "On the Inverse Kinematics, Statics, and Fault Tolerance of Cable-Suspended Robots" | 1998 | 🔴 | 🟡 | 🟢 | Necessary and sufficient equilibrium conditions for cable-suspended robots; left null space analysis. |

### Workspace Analysis (CDPM)

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Gouttefarde2006](sources/tendon_robots/Workspace_Analysis_%28CDPM%29/Gouttefarde2006.pdf) | Gouttefarde, M. & Gosselin, C. M. | "Analysis of the Wrench-Closure Workspace of Planar Parallel Cable-Driven Mechanisms" | 2006 | 🟡 | 🟢 | 🟢 | Wrench-closure workspace (WCW) definition; purely geometric analysis of planar 3-DOF + 4-cable CDPMs. |
| [Bosscher2006](sources/tendon_robots/Workspace_Analysis_%28CDPM%29/Bosscher2006.pdf) | Bosscher, P., Riechel, A. T. & Ebert-Uphoff, I. | "Wrench-Feasible Workspace Generation for Cable-Driven Robots" | 2006 | 🟡 | 🟢 | 🟢 | Wrench-feasible workspace (WFW) definition via available wrench set (zonotope) with bounded tension constraints. |
| [Stump2006](sources/tendon_robots/Workspace_Analysis_%28CDPM%29/Stump2006.pdf) | Stump, E. & Kumar, V. | "Workspaces of Cable-Actuated Parallel Manipulators" | 2006 | 🔴 | 🟡 | 🟢 | Closed-form workspace boundaries via convex analysis for cable-actuated parallel manipulators. |
| [Gouttefarde2011](sources/tendon_robots/Workspace_Analysis_%28CDPM%29/Gouttefarde2011.pdf) | Gouttefarde, M., Daney, D. & Merlet, J.-P. | "Interval-Analysis-Based Determination of the Wrench-Feasible Workspace of Parallel Cable-Driven Robots" | 2011 | 🔴 | 🟡 | 🟢 | Interval arithmetic for WFW computation accounting for geometric uncertainties. |
| [Lamaury2012](sources/tendon_robots/Workspace_Analysis_%28CDPM%29/Lamaury2012.pdf) | Lamaury, J. & Gouttefarde, M. | "A Tension Distribution Method with Improved Computational Efficiency" | 2012 | 🔴 | 🟡 | 🟢 | Efficient barycentric-style tension distribution for redundantly actuated CDPRs. Complements Pott2009 and Borgstrom2009 for force distribution. [DOI](https://doi.org/10.1007/978-3-642-31988-4_5) |
| [Wang2024TDCR](sources/tendon_robots/Workspace_Analysis_%28CDPM%29/Wang2024TDCR.pdf) | Wang, Y. et al. | "Workspace Analysis of Tendon-Driven Continuum Robots" | 2024 | 🔴 | 🟡 | 🟢 | Workspace analysis for tendon-driven continuum robots under actuation and geometric constraints. Context for TDCR reachability studies. |

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Walter2023](sources/tendon_robots/Conference_Papers/Walter2023.pdf) | Walter, J. et al. | "Tensegrity-Inspired Joint Can Protect from Impacts by Isolating" | 2023 | 🟢 | 🟢 | 🟢 | Introduces the tensegrity-inspired joint design used in this project's manipulator. Core reference for the mechanical concept. [DOI](https://doi.org/10.1007/978-3-031-32322-5_28) |
| [Walter2022](sources/tendon_robots/Conference_Papers/Walter2022.pdf) | Walter, J. et al. | "Bringing the New Class of CDPRs to Industrial CPS Based on RAMI 4.0" | 2022 | 🔴 | 🟡 | 🟢 | Contextualises cable-driven parallel robots within Industry 4.0 cyber-physical systems. Background for the industrial motivation. [DOI](https://doi.org/10.1109/ICPS51978.2022.9816877) |
| [Khosravi2011](sources/tendon_robots/Conference_Papers/Khosravi2011.pdf) | Khosravi, M. A. & Taghirad, H. D. | "Dynamic Analysis and Control of Cable-Driven Robots with Elastic Cables" | 2011 | 🔴 | 🟡 | 🟢 | Analyses cable elasticity effects on dynamics and proposes a robust controller. Relevant for modelling cable compliance in tension distribution. [DOI](https://doi.org/10.1109/CCECE.2011.6030586) |
| [Kirchhoff2016](sources/tendon_robots/Conference_Papers/Kirchhoff2016.pdf) | Kirchhoff, J. & von Stryk, O. | "Robust Trajectory Tracking Control for an Ultra-Lightweight Tendon-Driven Series Elastic Robot Arm" | 2016 | 🟡 | 🟡 | 🟢 | Robust trajectory tracking on an ultra-lightweight tendon-driven SEA arm. Reference for robust control in series-elastic tendon hardware. [DOI](https://doi.org/10.1109/AIM.2016.7576815) |
| [Picard2018](sources/tendon_robots/Conference_Papers/Picard2018.pdf) | Picard, E. et al. | "On the Control of a Cable-Driven Parallel Robot as a Haptic Master for Teleoperation" | 2018 | 🔴 | 🟡 | 🟢 | Control of a CDPR used as a haptic master for teleoperation. Application-side context for cable-driven control. [DOI](https://doi.org/10.1109/ICRA.2018.8460878) |

---

## Tendon-Driven Continuum Robots

### Review Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Rao2021](sources/tendon_robots/Review_Articles/Rao2021.pdf) | Rao, P., Peyron, Q., Lilge, S. & Burgner-Kahrs, J. | "How to Model Tendon-Driven Continuum Robots and Benchmark Modelling Performance" | 2021 | 🟡 | 🟢 | 🟢 | Benchmarks modelling approaches (constant-curvature, Cosserat rod, etc.) for tendon-driven continuum robots. Key context for understanding the limitations of simplified kinematic models. [DOI](https://doi.org/10.3389/frobt.2020.630245) |
| [Webster2010](sources/tendon_robots/Review_Articles/Webster2010.pdf) | Webster III, R. J. & Jones, B. A. | "Design and Kinematic Modeling of Constant Curvature Continuum Robots: A Review" | 2010 | 🔴 | 🟢 | 🟢 | Foundational review of constant-curvature continuum robot modelling. Provides the kinematic modelling context used to justify the piecewise-constant approach. [DOI](https://doi.org/10.1177/0278364910368147) |

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Nemoto2022](sources/tendon_robots/Journal_Articles/Nemoto2022.pdf) | Nemoto, T. et al. | "Highly Dynamic 2-DOF Cable-Driven Robotic Wrist Based on a Novel Topology" | 2022 | 🟢 | 🟢 | 🟢 | Presents the 2-DOF cable-driven wrist topology directly used in the tensegrity manipulator. Core hardware reference for the wrist mechanism. [DOI](https://doi.org/10.1109/LRA.2022.3159816) |
| [Miyasaka2020](sources/tendon_robots/Journal_Articles/Miyasaka2020.pdf) | Miyasaka, M. et al. | "Modeling Cable-Driven Robot With Hysteresis and Cable–Pulley Network Friction" | 2020 | 🟡 | 🟡 | 🟢 | Models hysteresis and friction in cable-pulley networks. Relevant for understanding sim-to-real discrepancies in tendon force transmission. [DOI](https://doi.org/10.1109/TMECH.2020.2973428) |
| [Kato2016](sources/tendon_robots/Journal_Articles/Kato2016.pdf) | Kato, T. et al. | "Tendon-driven continuum robot for neuroendoscopy: validation of extended kinematic mapping for hysteresis operation" | 2016 | 🔴 | 🟡 | 🟢 | Validates kinematic mapping with hysteresis compensation for tendon-driven continuum robots in medical applications. Background for hysteresis modelling. [DOI](https://doi.org/10.1007/s11548-015-1310-2) |
| [Saito2022](sources/tendon_robots/Journal_Articles/Saito2022.pdf) | Saito, N. & Morimoto, J. | "Characterization of Continuum Robot Arms Under Reinforcement Learning and Derived Improvements" | 2022 | 🟡 | 🟢 | 🟢 | Characterises continuum robot arms under RL control. Directly relevant as a prior-work comparison for RL-controlled tendon robots. [DOI](https://doi.org/10.3389/frobt.2022.895388) |
| Shahid2023 | Shahid, A. A. et al. | "Curriculum-Reinforcement Learning on Simulation Platform of Tendon-Driven High-DOF Underactuated Manipulator" | 2023 | 🟢 | 🟢 | 🟢 | Curriculum-based RL for tendon-driven underactuated manipulators. One of the closest prior works to this project's approach. [DOI](https://doi.org/10.3389/frobt.2023.1066518) |
| Cao2017 | Cao, K. et al. | "Workspace Analysis of Tendon-Driven Continuum Robots Based on Mechanical Interference Identification" | 2017 | 🔴 | 🟡 | 🟢 | Workspace analysis for TDCRs accounting for mechanical interference. Background for workspace analysis methodology. [DOI](https://doi.org/10.1115/1.4036395) |

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Nemoto2019 | Nemoto, T., Niiyama, R. & Iwata, H. | "Design and Analysis of a Compact Tendon-Driven Rotary Joint with Application to a Robotic Arm" | 2019 | 🟡 | 🟢 | 🟢 | Foundational control principle for tendon-driven orientation regulation used in the tensegrity manipulator's wrist mechanism. Cited as [84] in Klein (2023). |

### Theses

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Klein2023](sources/tendon_robots/Theses/Klein2023.pdf) | Klein, M. | *Arbeitsraumanalyse, Simulation und Bewegungsplanung eines seilgetriebenen robotischen Manipulators* | 2023 | 🟢 | 🟢 | 🟢 | **Primary reference.** Describes the tensegrity manipulator's kinematics, cable-tension distribution, PID control with gravity compensation, and step-response validation in Gazebo. The Isaac Sim implementation replicates control gains, test methodology, and NRMSE metrics from this thesis. |
| [Deutschmann2020](sources/tendon_robots/Theses/Deutschmann2020.pdf) | Deutschmann, B. | *Modeling and Control for a Class of Tendon-Driven Continuum Mechanisms* (PhD thesis) | 2020 | 🔴 | 🟢 | 🟢 | PhD thesis providing in-depth modelling and control theory for tendon-driven continuum mechanisms. Theoretical backbone for TDCR control. [DOI](https://doi.org/10.15488/10249) |
| [Ba2021](sources/tendon_robots/Theses/Ba2021.pdf) | Ba, W. | *Modelling, Analysis and Control of Tendon-Driven Continuum Robots* (PhD thesis, Univ. of Nottingham) | 2021 | 🔴 | 🟡 | 🟢 | PhD thesis covering modelling, analysis, and control methods for tendon-driven continuum robots. Broad reference for TDCR system design. |
| [SundhMasip2024](sources/tendon_robots/Theses/SundhMasip2024.pdf) | Sundh Masip, P. | *Design and Modelling of a Tendon-Driven Continuum Robot* (BSc thesis, KTH) | 2024 | 🔴 | 🟡 | 🟡 | Bachelor thesis on the design and modelling of a TDCR prototype. Practical reference for hardware-implementation choices. |

### Research Contributions

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Mukherjee](sources/tendon_robots/Research_Contributions/Mukherjee.pdf) | Mukherjee, S., Kamal, A. & Hirai, S. | Gravity compensation: τ_a = m·g·l_c·sin(θ) | — | 🟢 | 🟡 | 🟡 | Provides the gravity compensation formula used in the PID controller for step-response validation ([85] in Klein 2023, Eq. 3.2). |

---

## Cable-Driven Wrist Mechanisms

### Surveys

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Bajaj2019](sources/tendon_robots/Surveys/Bajaj2019.pdf) | Bajaj, N. M., Spiers, A. J. & Dollar, A. M. | "State of the Art in Artificial Wrists: A Review of Prosthetic and Robotic Wrist Design" | 2019 | 🔴 | 🟢 | 🟢 | Survey of 2/3-DOF wrist designs covering parallel vs. serial trade-offs and backdrivability. Contextualises the project's wrist topology. |

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Kim2018quaternion](sources/tendon_robots/Journal_Articles/Kim2018quaternion.pdf) | Kim, Y.-J., Kim, J.-I. & Jang, W. | "Quaternion Joint: Dexterous 3-DOF Joint Representing Quaternion Motion for High-Speed Safe Interaction" | 2018 | 🔴 | 🟡 | 🟢 | 3-DOF wrist with 2-DOF spherical rolling + wire actuation; zero singularities. Alternative wrist topology reference. |
| [Jiang2019](sources/tendon_robots/Journal_Articles/Jiang2019.pdf) | Jiang, H., Xiao, C., Li, J., Zhong, Z., Zhang, T. & Guan, Y. | "Design and Modeling of a 2-DOF Cable-Driven Parallel Wrist Mechanism" | 2019 | 🔴 | 🟡 | 🟢 | 2-DOF hemispherical cable-driven wrist with no singular points and 1.07 × 10⁶ Nmm/rad stiffness. |
| [Tsumaki2018](sources/tendon_robots/Journal_Articles/Tsumaki2018.pdf) | Tsumaki, Y., Suzuki, Y., Sasaki, N., Obara, E. & Kanazawa, S. | "A 7-DOF Wire-Driven Lightweight Arm with Wide Wrist Motion Range" | 2018 | 🔴 | 🟡 | 🟢 | 7-DOF arm with a 3-DOF parallel-wire wrist and novel wire arrangements. |
| [Tsumaki2014](sources/tendon_robots/Journal_Articles/Tsumaki2014.pdf) | Tsumaki, Y., Shimanuki, S., Ono, F. & Han, H. T. | "Ultra-Lightweight Forearm with a Parallel-Wire Mechanism" | 2014 | 🔴 | 🟡 | 🟢 | Early parallel-wire forearm design. Historical context for cable-driven wrist development. |

---

## RL for Tendon-Driven Robots

### Journal & Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Or2023](sources/tendon_robots/Journal_%26_Conference_Papers/Or2023.pdf) | Or, K., Wu, K. Y., Nakano, K., Ikeda, M., Ando, M., Kuniyoshi, Y. & Niiyama, R. | "Curriculum-reinforcement learning on simulation platform of tendon-driven high-degree of freedom underactuated manipulator" | 2023 | 🟢 | 🟢 | 🟢 | Robostrich 18-DOF tendon arm trained with SAC and curriculum learning in MuJoCo. Key prior work for tendon-driven RL manipulation. |
| [Guist2024](sources/tendon_robots/Journal_%26_Conference_Papers/Guist2024.pdf) | Guist, S., Schneider, J., Ma, H. et al. | "Safe & Accurate at Speed with Tendons: A Robot Arm for Exploring Dynamic Motion" | 2024 | 🟡 | 🟢 | 🟢 | PAMY2 4-DOF pneumatic tendon arm trained over 25 days of real RL for dynamic table tennis. Demonstrates real-world tendon RL feasibility. |
| [Liu2024](sources/tendon_robots/Journal_%26_Conference_Papers/Liu2024.pdf) | Liu, Y., Luo, Z., Qian, S., Wang, S. & Wu, Z. | "Deep Reinforcement Learning and Decoupling PID Control of a Humanoid Cable-Driven Hybrid Robot" | 2024 | 🟡 | 🟢 | 🟢 | Cable-driven humanoid arm with decoupling PID + TD3, achieving 77.58 % error reduction. Hybrid RL/classical control. |
| [Islam2024](sources/tendon_robots/Journal_%26_Conference_Papers/Islam2024.pdf) | Islam, S., He, Z. & Ciocarlie, M. | "Task-Based Design and Policy Co-Optimization for Tendon-driven Underactuated Kinematic Chains" | 2024 | 🟡 | 🟢 | 🟢 | 3-link 2-actuator MORPH framework for design-control co-optimization with sim-to-real transfer. |
| [Yuryev2026](sources/tendon_robots/Journal_%26_Conference_Papers/Yuryev2026.pdf) | Yuryev, V. & Hughes, J. | "Tendon Force Modeling for Sim2Real Transfer of Reinforcement Learning Policies for Tendon-Driven Robots" | 2026 | 🟡 | 🟢 | 🟢 | Transformer-based tendon force model reducing sim-to-real gap by 41 % and improving pose tracking by 50 %. |
| [Elstner2024](sources/tendon_robots/Journal_%26_Conference_Papers/Elstner2024.pdf) | Elstner, L., Kyrkjebø, E. & Stoelen, M. F. | "Contact-Rich Task Learning on an Articulated Soft Robot Arm Through Simulation" | 2024 | 🟡 | 🟡 | 🟢 | Antagonistic VSA with stretchable tendons in MuJoCo RL. Context for soft-arm RL. |
| [Eberlein2025](sources/tendon_robots/Journal_%26_Conference_Papers/Eberlein2025.pdf) | Eberlein, M. et al. | "ORCA: An Open-Source, Reliable, Cost-Effective, Anthropomorphic Robotic Hand for Uninterrupted Dexterous Task Learning" | 2025 | 🟡 | 🟢 | 🟢 | 17-DOF tendon-driven hand trained with A2C in IsaacGymEnvs; zero-shot sim-to-real transfer. |
| [Toshimitsu2023](sources/tendon_robots/Journal_%26_Conference_Papers/Toshimitsu2023.pdf) | Toshimitsu, Y., Forrai, B., Cangan, B. G. et al. | "Getting the Ball Rolling: Learning a Dexterous Policy for a Biomimetic Tendon-Driven Hand with Rolling Contact Joints" | 2023 | 🟡 | 🟡 | 🟢 | Faive Hand tendon-driven in-hand manipulation via A2C with asymmetric observations. |
| [Theodorou2012](sources/tendon_robots/Journal_%26_Conference_Papers/Theodorou2012.pdf) | Theodorou, E., Buchli, J. & Schaal, S. | "Tendon-Driven Variable Impedance Control Using Reinforcement Learning" | 2012 | 🔴 | 🟡 | 🟢 | PI² RL on ACT hand for antagonist co-contraction learning. Early tendon RL reference. |
| [Marjaninejad2019](sources/tendon_robots/Journal_%26_Conference_Papers/Marjaninejad2019.pdf) | Marjaninejad, A., Urbina-Meléndez, D. & Valero-Cuevas, F. J. | "Autonomous Functional Movements in a Tendon-Driven Limb via Limited Experience" | 2019 | 🔴 | 🟡 | 🟢 | G2P algorithm for 2-joint 3-tendon limb control via motor babbling. Data-efficient tendon learning. |
| [Li2025robust](sources/tendon_robots/Journal_%26_Conference_Papers/Li2025robust.pdf) | Li, X. et al. | "Robust Reinforcement Learning Control for Tendon-Driven Robots" | 2025 | 🟡 | 🟢 | 🟢 | Robust RL control approach tailored to tendon-driven manipulators with parameter uncertainty. Direct comparator for the project's RL pipeline. |
| [Maghooli2025](sources/tendon_robots/Journal_%26_Conference_Papers/Maghooli2025.pdf) | Maghooli, M. et al. | "Adaptive Control of Cable-Driven Robotic Manipulators with Time-Varying Cable Stiffness" | 2025 | 🟡 | 🟡 | 🟢 | Adaptive controller accounting for time-varying cable stiffness in CDPRs. Relevant for handling cable compliance drift in long-running deployments. |
