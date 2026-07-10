"""DLS-IK + inner-PID heuristic for the physical *hierarchical* reach variant.

This is the physical-variant analogue of the PD ``HeuristicIKAgent``
(``evaluate_reach.py``).  Because the hierarchical action term
(:class:`~tensegrity_pick.robots.HierarchicalPhysicalTendonAction`) already
consumes joint-space **set-points** and regulates the cable tensions with the
validated inner PID loop, the heuristic only has to produce good joint targets:

1. Solve **position-priority damped-least-squares IK** for the effective 5-DOF
   arm (base-Y, base-Z, elbow, wrist-Y, wrist-X).
2. Emit the solution through the hierarchical action's set-point convention;
   the inner PID→tension controller tracks it on the physical 4-bar robot.

Why the FK reference robot for IK
---------------------------------
The physical robot's PhysX Jacobian is over its seven *tree* joints
(base ×2, rod_left, rod_right, coupler_left, wrist ×2); the elbow is a
closed-chain 4-bar whose three linkage joints are coupled, so a differential-IK
step on that Jacobian fights the loop-closure constraint.  The scene already
carries a hidden, collision-disabled **PD reference robot** with a single clean
``elbow_joint`` and identical link geometry (it is what the target sampler uses
for FK).  We run the IK on that reference model — via the same
write-joint → ``update_articulations_kinematic`` → ``get_link_transforms``
kinematic-FK path the sampler uses — and feed the resulting effective-elbow /
wrist / base angles to the physical robot as set-points.  The reference robot is
saved and restored each call, so target sampling is unaffected.

The effective elbow angle of the physical robot (forearm twist about the
upper-arm X axis, :func:`compute_lower_arm_angle_and_rate`) is exactly the
reference robot's ``elbow_joint`` angle, so the mapping is 1:1.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

from tensegrity_pick.robots import compute_lower_arm_angle_and_rate

# Effective 5-DOF arm, in the order the IK state vector uses.
_REF_JOINT_NAMES = [
    "base_y_joint", "base_z_joint", "elbow_joint", "wrist_y_joint", "wrist_x_joint",
]


class HeuristicHierarchicalIKAgent:
    """Scripted IK baseline for ``Template-Reach-Tensegrity-Physical-Hierarchical*``.

    Parameters
    ----------
    ik_iters : max damped-least-squares iterations per control step (early-exits
        once the position error is below ``pos_tol``).  Seeding each step at the
        physical robot's current configuration means later steps converge in a
        few iterations.
    damping : DLS damping λ [m]; ``J Jᵀ + λ²I`` is inverted.
    dq_clip : per-iteration joint step clamp [rad / m] (stability).
    pos_tol : early-exit Cartesian tolerance [m].
    """

    def __init__(self, env, ik_iters: int = 20, damping: float = 0.05,
                 dq_clip: float = 0.25, pos_tol: float = 0.005):
        u = env.unwrapped
        self.device = u.device
        self.num_envs = u.num_envs
        self.ik_iters = int(ik_iters)
        self.damping = float(damping)
        self.dq_clip = float(dq_clip)
        self.pos_tol = float(pos_tol)
        self._delta = 1.0e-3  # finite-difference perturbation

        # Command ramp rates per control step (30 Hz), applied to the EMITTED
        # command — the IK solution is a *goal*, not a step command.  Without
        # this the base (PD stiffness 8000) is slammed across its range in one
        # step; the resulting accelerations shake the hanging four-bar hard
        # enough to strain/branch-flip the loop closure (measured 15 % broken
        # steps before this ramp).  [base_y m, base_z m, elbow, wrist_y,
        # wrist_x rad] per second:
        self._cmd_rates = torch.tensor(
            [0.4, 0.4, 1.0, 3.0, 3.0], device=self.device
        )
        self._cmd: torch.Tensor | None = None  # ramped command state (B, 5)

        self.robot = u.scene["robot"]

        # FK reference robot (the hidden PD model used for target sampling).
        cmd_term = u.command_manager.get_term("ee_pose")
        ref_name = cmd_term.cfg.fk_reference_asset_name
        if ref_name is None:
            raise SystemExit(
                "[heuristic] the hierarchical physical variant must have an FK "
                "reference robot; none configured on the ee_pose command term."
            )
        self.fk_robot = u.scene[ref_name]
        self.fk_body_idx = self.fk_robot.find_bodies(cmd_term.cfg.fk_reference_body_name)[0][0]
        self.ref_jids = self.fk_robot.find_joints(_REF_JOINT_NAMES, preserve_order=True)[0]

        # Physical robot handles for seeding the IK at the current config.
        self.phys_base_ids = self.robot.find_joints(["base_y_joint", "base_z_joint"], preserve_order=True)[0]
        self.phys_wrist_ids = self.robot.find_joints(["wrist_y_joint", "wrist_x_joint"], preserve_order=True)[0]
        self.root_idx = self.robot.find_bodies("root_link")[0][0]
        self.forearm_idx = self.robot.find_bodies("forearm_link")[0][0]

        # Reference-robot joint limits (finite fallback for any invalid ones).
        rsoft = self.fk_robot.data.soft_joint_pos_limits[0, self.ref_jids].clone()
        lo, hi = rsoft[:, 0], rsoft[:, 1]
        default = self.fk_robot.data.default_joint_pos[0, self.ref_jids]
        bad = ~torch.isfinite(lo) | ~torch.isfinite(hi) | ((hi - lo) <= 1e-5) | ((hi - lo) > 8 * torch.pi)
        lo = torch.where(bad, default - 0.75, lo)
        hi = torch.where(bad, default + 0.75, hi)
        self.ref_lo, self.ref_hi = lo, hi  # (5,)

        # Action-inversion parameters.
        # base_action: JointPositionToLimits over the physical base joints.
        base_soft = self.robot.data.soft_joint_pos_limits[:, self.phys_base_ids]  # (N, 2, 2)
        self.base_lower = base_soft[..., 0]
        self.base_upper = base_soft[..., 1]
        # arm_tendon: set-point affine map, from the action cfg.
        arm_cfg = u.action_manager.get_term("arm_tendon").cfg
        self.elbow_lo, self.elbow_hi = arm_cfg.elbow_setpoint_range
        self.wrist_lo, self.wrist_hi = arm_cfg.wrist_setpoint_range

    def reset(self, env_ids=None):
        pass

    # ── kinematic FK on the reference robot ─────────────────────────────────
    def _fk_ee_pos(self, q: torch.Tensor) -> torch.Tensor:
        """EE world position for effective config ``q`` (N, 5)."""
        self.fk_robot.write_joint_position_to_sim(q, joint_ids=self.ref_jids)
        self.fk_robot.data._physics_sim_view.update_articulations_kinematic()
        lt = self.fk_robot.data._root_physx_view.get_link_transforms()
        return lt[:, self.fk_body_idx, :3].clone()

    # ── one control-step action ─────────────────────────────────────────────
    def act(self, env) -> torch.Tensor:
        u = env.unwrapped

        # Target in world frame (command is in the controlled robot's root frame).
        cmd_b = u.command_manager.get_command("ee_pose")
        target_pos_w, _ = math_utils.combine_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, cmd_b[:, :3], cmd_b[:, 3:7],
        )

        # Seed at the physical robot's current effective configuration.
        base_q = self.robot.data.joint_pos[:, self.phys_base_ids]
        elbow_q, _ = compute_lower_arm_angle_and_rate(self.robot, self.root_idx, self.forearm_idx)
        wrist_q = self.robot.data.joint_pos[:, self.phys_wrist_ids]
        q = torch.cat([base_q, elbow_q.unsqueeze(-1), wrist_q], dim=-1)  # (N, 5)
        q = torch.max(torch.min(q, self.ref_hi), self.ref_lo)

        # Save the reference robot state; the IK sweep perturbs it.
        saved_pos = self.fk_robot.data.joint_pos.clone()
        saved_vel = self.fk_robot.data.joint_vel.clone()

        eye3 = torch.eye(3, device=self.device)
        for _ in range(self.ik_iters):
            ee = self._fk_ee_pos(q)
            err = target_pos_w - ee                                   # (N, 3)
            if float(err.norm(dim=-1).max()) < self.pos_tol:
                break
            # Finite-difference position Jacobian (N, 3, 5).
            J = torch.zeros(self.num_envs, 3, 5, device=self.device)
            for j in range(5):
                # perturb away from the nearer limit so the clamp never zeroes it
                near_hi = q[:, j] > (self.ref_hi[j] - self._delta)
                sign = torch.where(near_hi, -1.0, 1.0)
                qp = q.clone()
                qp[:, j] = qp[:, j] + sign * self._delta
                J[:, :, j] = (self._fk_ee_pos(qp) - ee) / (sign.unsqueeze(-1) * self._delta)
            # DLS step: dq = Jᵀ (J Jᵀ + λ²I)⁻¹ err
            JT = J.transpose(1, 2)
            A = J @ JT + (self.damping ** 2) * eye3
            dq = (JT @ torch.linalg.solve(A, err.unsqueeze(-1))).squeeze(-1)
            dq = dq.clamp(-self.dq_clip, self.dq_clip)
            q = torch.max(torch.min(q + dq, self.ref_hi), self.ref_lo)

        # Diagnostics: residual of the IK solution on the reference model, plus
        # the previous commanded solution vs. the physical robot's achieved
        # config (real per-DOF tracking error of the inner controller).
        ik_ee = self._fk_ee_pos(q)
        self.last_ik_residual = (target_pos_w - ik_ee).norm(dim=-1)
        achieved = torch.cat([base_q, elbow_q.unsqueeze(-1), wrist_q], dim=-1)
        if getattr(self, "last_solution", None) is not None:
            self.last_track_err = (self.last_solution - achieved).abs()
        self.last_solution = q.clone()
        self.last_command = q.clone()
        self.last_achieved = achieved.clone()

        # Restore the reference robot (target sampling must be unaffected).
        self.fk_robot.write_joint_position_to_sim(saved_pos)
        self.fk_robot.write_joint_velocity_to_sim(saved_vel)
        self.fk_robot.data._physics_sim_view.update_articulations_kinematic()
        self.fk_robot.data._body_link_pose_w.timestamp = -1

        # ── Command ramping ────────────────────────────────────────────────
        # Step the emitted command toward the IK goal at bounded rates; on
        # per-env reset (episode_length == 0) re-seed the ramp at the robot's
        # current (default) configuration.
        step_dt = float(u.step_dt)
        if self._cmd is None:
            self._cmd = achieved.clone()
        just_reset = u.episode_length_buf <= 1
        if just_reset.any():
            self._cmd[just_reset] = achieved[just_reset]
        max_step = self._cmd_rates * step_dt
        self._cmd += (q - self._cmd).clamp(-max_step, max_step)
        q_cmd = self._cmd

        # Map the ramped command into the hierarchical action vector
        # [base_y, base_z, elbow_sp, wrist_y_sp, wrist_x_sp].
        base_clamped = torch.max(torch.min(q_cmd[:, :2], self.base_upper), self.base_lower)
        base_a = math_utils.scale_transform(base_clamped, self.base_lower, self.base_upper)
        elbow_a = (2.0 * (q_cmd[:, 2] - self.elbow_lo) / (self.elbow_hi - self.elbow_lo) - 1.0).clamp(-1, 1)
        wy_a = (2.0 * (q_cmd[:, 3] - self.wrist_lo) / (self.wrist_hi - self.wrist_lo) - 1.0).clamp(-1, 1)
        wx_a = (2.0 * (q_cmd[:, 4] - self.wrist_lo) / (self.wrist_hi - self.wrist_lo) - 1.0).clamp(-1, 1)
        return torch.cat(
            [base_a, elbow_a.unsqueeze(-1), wy_a.unsqueeze(-1), wx_a.unsqueeze(-1)], dim=-1
        ).clamp(-1.0, 1.0)
