// §2.3 Workspace Analysis and Dexterity Metrics
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure

== Workspace Analysis and Dexterity Metrics <sec:workspace_analysis_sampling_metrics>

#reg-sym("sym:w", "sym:J_mat", "sym:kappa")

The workspace of a serial manipulator is the set of all end-effector poses that are reachable under joint limits and kinematic constraints. Workspace analysis is used to verify that a robot can reach task-relevant regions and to identify configurations that should be avoided due to singularities or poor motion quality.~@Rastegar1990WorkspaceMonteCarlo@Dong2013WorkspaceDensity

=== Forward Kinematics and Sampling Motivation

#ac("FK") is defined as the mapping from a joint configuration $bold(q)$ to an end-effector pose, i.e., position and orientation of a designated tool frame. For a general serial chain, this mapping can be written as

$ bold(T)_"EE" = f(bold(q)), $ <eq:fk_mapping>

where $bold(T)_"EE" in S E(3)$ denotes a homogeneous transformation of the end-effector frame. In many practical systems, analytical derivation of workspace boundary surfaces is unsolvable due to high #ac("DoF"), joint coupling, or non-standard kinematics. In such cases, the workspace is frequently approximated numerically by sampling joint space and evaluating #ac("FK") repeatedly. The Monte Carlo approach is widely used because it is simple to implement, directly incorporates joint limits, and does not require explicit boundary derivation.~@Rastegar1990WorkspaceMonteCarlo

=== Monte Carlo Workspace Sampling and Voxel Aggregation

In the Monte Carlo approach, configurations are sampled from a established distribution over the defined joint ranges. In the simplest implementation, each joint $q_i$ is sampled independently and uniformly within its limit interval. For each sample $bold(q)^((k))$, the corresponding end-effector pose is computed by #ac("FK"). The resulting set of poses is then aggregated into a discrete spatial representation, e.g., Cartesian voxels, to estimate reachability statistics per region. This computation is conceptually aligned with established robotics software workflows that generate reachable workspace point sets paired with joint configurations~@mathworks_generaterobotworkspace.

When voxel aggregation is used, the workspace is partitioned into a regular grid of bins (voxels) in position and optionally orientation space. Each sample contributes to the occupancy count of the voxel containing the sampled pose. This discretization supports additional metrics beyond mere reachability, including density-based indicators of how many joint-space configurations map to a given region.~@Dong2013WorkspaceDensity

=== Dexterity Metrics

==== Yoshikawa manipulability
Dexterity is commonly evaluated using Jacobian-based indices. Let $J(bold(q))$ denote the geometric Jacobian that maps joint velocities to end-effector twist. The Yoshikawa manipulability measure is defined as

$ w(bold(q)) = sqrt(det(J(bold(q)) J(bold(q))^top)). $ <eq:yoshikawa_manipulability>

This scalar is proportional to the volume of the manipulability ellipsoid and becomes zero at kinematic singularities where the Jacobian loses rank. Higher values indicate that small joint motions can produce motion in a wider range of Cartesian directions, which is desirable for tasks requiring dexterous repositioning and local adjustments.~@Yoshikawa1985Manipulability

==== Inverse condition number as an isotropy index
A complementary measure is derived from the Jacobian condition number. Salisbury and Craig introduced the use of Jacobian conditioning to identify well-conditioned (isotropic) operating points and to quantify error amplification through the kinematic mapping. Let $sigma_min$ and $sigma_max$ denote the minimum and maximum singular values of $J(bold(q))$. The inverse condition number is defined as

$ kappa_"inv" (bold(q)) = sigma_min (J(bold(q))) / sigma_max (J(bold(q))). $ <eq:inv_condition_number>

This index lies in $[0,1]$, approaches zero near singular configurations, and equals one at perfectly isotropic configurations (equal singular values), where motion/force transmission properties are directionally uniform.~@Salisbury1982ArticulatedHands

==== Reachability density
Reachability density is defined here as the number of sampled configurations whose end-effector poses fall within a given voxel. Under uniform joint sampling, this count is proportional to the joint-space volume that maps into the voxel region. Thus, higher density indicates that more distinct configurations can realize poses in that region. This concept is closely related to workspace density formulations that assign each voxel the number of reachable points (or a normalized probability density) and interpret higher density as higher positional/orientational reach richness and, in discretely actuated systems, higher achievable pose resolution.~@Dong2013WorkspaceDensity

=== Implications for Task-Space Design in Learning-Based Manipulation

For manipulation tasks in cluttered scenes and for #ac("RL")-based control, the workspace and dexterity metrics directly inform environment layout and action-space design. Task objects (e.g., pick targets, placement zones) are preferably positioned in regions with high reachability density and consistently high manipulability to reduce the probability that the robot is driven into near-singular configurations during exploration. Similarly, action bounds for end-effector motion primitives can be restricted to dexterous regions to improve learning stability and to reduce large corrective motions caused by kinematic ill-conditioning.~@Yoshikawa1985Manipulability@Salisbury1982ArticulatedHands@Dong2013WorkspaceDensity

When tensegrity compliance is present, these considerations become more important because poor kinematic conditioning can amplify transmission uncertainties into larger Cartesian errors, which in turn increases undesired contact forces during object interaction~@Miyasaka2020HysteresisFriction@Bicchi2004FastSoftArm.
