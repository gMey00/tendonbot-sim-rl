// Appendices
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *
#import "../../../shared/formatting/template.typ": faps-table

// Reset heading numbering for appendix style
#set heading(numbering: "A.1")
#counter(heading).update(0)

= Appendix: Simulation and Control Parameters <appendix:validation_params>

== Joint Drive Parameters <appendix:drive_params>

@tab:drive_params lists the actuator drive parameters for the tensegrity manipulator's joints.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Joint Group*], [*Stiffness*], [*Damping*], [*Effort Limit*], [*Vel. Limit*]),
    [`base_y`], [8~000], [800], [300~N], [5.0~m/s],
    [`base_z`], [8~000], [800], [200~N], [5.0~m/s],
    [`elbow`], [400], [20], [40~N·m], [1.0~rad/s],
    [`wrist` (both)], [400], [20], [10~N·m], [0.5~rad/s],
  ),
  caption: [Implicit actuator PD drive parameters for the tensegrity manipulator.],
  short-caption: [Tensegrity joint drive parameters],
) <tab:drive_params>

== Step-Response PID Parameters <appendix:step_pid>

@tab:step_pid lists the PID parameters used in the step-response validation tests (@sec:model_validation). Gains are scaled $approx 133 times$ from Klein's originals to match Isaac Sim's effective rotational inertia.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Joint*], [$K_p$], [$K_i$], [$K_d$], [*Saturation (N)*]),
    [`elbow_joint`], [40], [4], [3], [400],
    [`wrist_y_joint`], [10], [2], [0.3], [600],
    [`wrist_x_joint`], [10], [2], [0.3], [600],
  ),
  caption: [PID gains and torque saturation for step-response testing.],
  short-caption: [Step-response PID gains],
) <tab:step_pid>

The PID controllers include feedforward gravity compensation: $tau_g = m_e dot g dot l_c dot sin(theta)$.

= Appendix: PPO Hyperparameters <appendix:ppo_hyperparams>

@tab:ppo_full compares the full PPO configurations for the reach and cube place tasks.

#faps-table(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Parameter*], [*Reach*], [*Cube Place*]),
    [Seed], [42], [42],
    [Network architecture], [$[64, 64]$ ELU], [$[256, 128, 64]$ ELU],
    [Shared policy/value], [Yes], [Yes],
    [Rollout horizon], [24 steps], [48 steps],
    [Learning epochs], [5], [5],
    [Mini-batches], [4], [8],
    [Learning rate], [$10^(-3)$], [$10^(-4)$],
    [LR scheduler], [KL-adaptive], [KL-adaptive],
    [KL threshold], [0.01], [0.004],
    [Discount $gamma$], [0.99], [0.99],
    [GAE $lambda$], [0.95], [0.95],
    [Clip ratio $epsilon$], [0.2], [0.2],
    [Value clip], [0.2], [0.2],
    [Entropy scale], [0.01], [0.01],
    [Value loss scale], [1.0], [1.0],
    [Grad norm clip], [1.0], [1.0],
    [Initial log-std range], [$[-20, 2]$], [$[-2.5, 2]$],
    [State preprocessor], [RunningStandardScaler], [RunningStandardScaler],
    [Total timesteps], [48~000], [500~000],
    [`num_envs`], [4~096], [4~096],
  ),
  caption: [Full PPO hyperparameter comparison between reach and cube place tasks.],
  short-caption: [PPO hyperparameter comparison],
) <tab:ppo_full>

= Appendix: Cube Place Reward Terms <appendix:place_regularization>

@tab:place_full_rewards lists all reward terms used in the cube place task, including regularization and metric-only terms.

#faps-table(
  table(
    columns: (auto, auto, 1fr, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Term*], [*Weight*], [*Description*], [*Gate*]),
    table.hline(),
    table.cell(colspan: 4)[_Task Rewards_],
    [`reaching_object`], [$+2.0$], [Fingertip-to-cube distance shaping], [—],
    [`grasping`], [$+5.0$], [Closure $times$ proximity], [—],
    [`lifting_object`], [$+3.0$], [Binary: above belt, near gripper], [—],
    [`height_bonus`], [$+5.0$], [$min(Delta z, 0.30) / 0.30$], [—],
    [`object_velocity_penalty`], [$+5.0$], [Penalize excessive cube motion], [—],
    [`goal_tracking`], [$+40.0$], [$1 - tanh(d_"xy" / 1.0)$], [`was_grasped`],
    [`goal_tracking_fine`], [$+10.0$], [$1 - tanh(d_"xy" / 0.20)$], [`was_grasped`],
    [`release`], [$+25.0$], [$(1 - "cl.") times "in"_"xy" times "above"$], [`was_grasped`],
    [`green_in_target`], [$+100.0$], [Cube center inside drum volume], [`was_grasped`],
    [`red_in_target`], [$-12.0$], [Red cube in drum (penalty)], [red active],
    table.hline(),
    table.cell(colspan: 4)[_Regularization_],
    [`object_stay_bonus`], [$+3.0$], [Bonus for cube remaining on belt], [—],
    [`action_rate`], [$-10^(-4) arrow -2 times 10^(-3)$], [L2 action delta (curriculum)], [—],
    [`joint_vel`], [$-10^(-4) arrow -2 times 10^(-3)$], [L2 joint velocity (curriculum)], [—],
    [`fingertip_orientation`], [$+0.5$], [Fingertip alignment shaping], [—],
    [`belt_contact`], [$-10.0$], [Finger tips below belt surface], [—],
    [`joint_torque`], [$-0.05$], [L2 torque penalty], [—],
    [`cube_off_conveyor`], [$-5.0$], [Cube knocked off belt], [—],
    table.hline(),
    table.cell(colspan: 4)[_Metric-Only (Small Weight)_],
    [`cube_height_info`], [$+0.01$], [Cube height above belt], [—],
    [`gripper_state_info`], [$+0.01$], [Gripper closure tracking], [—],
    [`collision_penalty_info`], [$-0.01$], [Self-collision monitoring], [—],
  ),
  caption: [Complete cube place reward specification. Curriculum terms ramp weights at 200k training steps.],
  short-caption: [Cube place reward specification],
) <tab:place_full_rewards>
