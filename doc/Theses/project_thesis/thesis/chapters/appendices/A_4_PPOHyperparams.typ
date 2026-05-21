// Appendix A.2 — #ac("PPO") Hyperparameters
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-table
#reg-sym("sym:epsilon", "sym:gamma")

= Appendix: #ac("PPO") Hyperparameters <appendix:ppo_hyperparams>

@tab:ppo_full compares the full #ac("PPO") configurations for the reach and cube place tasks.

#faps-table(
  table(
    columns: (auto, auto, auto),
    table.header([*Parameter*], [*Reach*], [*Cube Place*]),
    [Seed], [42], [42],
    [Network architecture], [$[64, 64]$ #acs("ELU")], [$[256, 128, 64]$ #acs("ELU")],
    [Shared policy/value], [Yes], [Yes],
    [Rollout horizon], [24 steps], [128 steps],
    [Learning epochs], [5], [5],
    [Mini-batches], [4], [4],
    [Learning rate], [$10^(-3)$], [$10^(-4)$],
    [#acs("LR") scheduler], [#acs("KL")-adaptive], [#acs("KL")-adaptive],
    [#acs("KL") threshold], [0.01], [0.004],
    [Discount $gamma$], [0.99], [0.99],
    [#ac("GAE") $lambda$], [0.95], [0.95],
    [Clip ratio $epsilon$], [0.2], [0.2],
    [Value clip], [0.2], [0.2],
    [Entropy scale], [0.01], [0.005],
    [Value loss scale], [1.0], [2.0],
    [Reward shaper scale], [1.0], [0.5],
    [Grad norm clip], [1.0], [1.0],
    [Initial log-std $sigma_0$], [0.0], [0.0],
    [Log-std clip range], [$[-20, 2]$], [$[-2.5, 2]$],
    [State preprocessor], [RunningStandardScaler], [RunningStandardScaler],
    [Value preprocessor], [RunningStandardScaler], [RunningStandardScaler],
    [Total timesteps], [48~000], [300~000],
    [Parallel environments (`num_envs`)], [4~096], [1~024],
  ),
  caption: [Full #ac("PPO") hyperparameter comparison between reach and cube place tasks.],
  short-caption: [#ac("PPO") hyperparameter comparison],
) <tab:ppo_full>
