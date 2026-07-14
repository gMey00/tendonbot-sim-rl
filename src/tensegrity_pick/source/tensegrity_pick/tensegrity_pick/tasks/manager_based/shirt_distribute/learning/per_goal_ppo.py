"""PPO with per-goal return normalization + per-goal advantage normalization.

Implements recommendation #1 of
``doc/reports/tmp/RESEARCH_REPORT_goal_conditioned_mode_collapse`` §C for the
shirt_distribute goal-conditioned mode collapse: the single aggregate
``RunningStandardScaler`` on value targets, together with the global advantage
normalization in stock GAE, couples the three goals so the hardest-so-far goal's
*normalized* advantage is driven systematically negative and a unimodal shared
actor is pushed away from it (winner-take-all / per-seed 2-of-3 specialization).

This subclass changes exactly two things versus stock skrl PPO, leaving every
hyperparameter untouched so the effect is attributable:

  1. **Per-goal value/return normalization** (PopArt-style; Hessel et al. 2019).
     One ``RunningStandardScaler`` per commanded bin replaces the single shared
     one.  The commanded bin of each transition is recovered from the goal
     one-hot in the last ``NUM_BINS`` observation dims.
  2. **Per-goal advantage normalization.**  GAE advantages are standardized
     *within each bin group* instead of across the whole batch, so a goal whose
     returns are transiently lower no longer inherits a negative mean advantage
     from the easy goal.

Pair with :class:`GoalMultiHeadValue` (per-goal value heads) — the two together
are the report's bundled #1 intervention.

The stock ``value_preprocessor`` must be disabled in the agent cfg
(``value_preprocessor: null``) — this agent owns value normalization.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.ppo import PPO
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

# Exactly three discrete goals (reusable / recyclable / trash); the goal one-hot
# occupies the last NUM_BINS observation dims (mdp.rewards.target_bin_onehot).
NUM_BINS = 3


class PerGoalPPO(PPO):
    """PPO variant with per-bin value + advantage normalization."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # One value/return normalizer per commanded bin. Registered for
        # checkpointing so resume/eval restore per-goal running statistics.
        self._pg_value_pre = torch.nn.ModuleList(
            [RunningStandardScaler(size=1, device=self.device) for _ in range(NUM_BINS)]
        )
        for b in range(NUM_BINS):
            self.checkpoint_modules[f"pg_value_preprocessor_{b}"] = self._pg_value_pre[b]

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _bins_from_obs(observations: torch.Tensor) -> torch.Tensor:
        """Recover commanded bin per row from the trailing goal one-hot.

        Argmax over the one-hot columns is invariant to per-column running
        standardization (columns share stats under uniform sampling), so this is
        robust whether ``observations`` are raw or preprocessed."""
        return observations[..., -NUM_BINS:].argmax(dim=-1)

    def _pg_value(
        self, values: torch.Tensor, bins: torch.Tensor, *, inverse: bool = False, train: bool = False
    ) -> torch.Tensor:
        """Apply the per-bin value normalizer to ``values`` (…, 1) using ``bins``.

        ``bins`` broadcasts against ``values`` on all but the last (size-1) dim.
        """
        out = torch.empty_like(values)
        flat_bins = bins.reshape(-1)
        flat_vals = values.reshape(-1, values.shape[-1])
        flat_out = out.reshape(-1, values.shape[-1])
        for b in range(NUM_BINS):
            mask = flat_bins == b
            if mask.any():
                flat_out[mask] = self._pg_value_pre[b](flat_vals[mask], inverse=inverse, train=train)
        return out

    # -------------------------------------------------------------------- act
    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ):
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
            self._current_log_prob = outputs["log_prob"]

            if self.training:
                values, _ = self.value.act(inputs, role="value")
                bins = self._bins_from_obs(observations)
                # De-normalize per goal to store real-scale values for GAE.
                self._current_values = self._pg_value(values, bins, inverse=True)

        return actions, outputs

    # ----------------------------------------------------------------- update
    def update(self, *, timestep: int, timesteps: int) -> None:
        # bootstrap value of the last next-observation, per goal
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_bins = self._bins_from_obs(self._current_next_observations)
            last_values = self._pg_value(last_values, last_bins, inverse=True)

        # per-transition commanded bin (from stored raw observations)
        observations = self.memory.get_tensor_by_name("observations")
        bins = self._bins_from_obs(observations)  # (memory_size, num_envs)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = self._compute_gae_per_goal(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            bins=bins,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
        )

        # per-goal value/return normalization (replaces the aggregate scaler)
        self.memory.set_tensor_by_name("values", self._pg_value(values, bins, train=True))
        self.memory.set_tensor_by_name("returns", self._pg_value(returns, bins, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # --- diagnostics: per-goal advantage means confirm the mechanism -----
        with torch.no_grad():
            flat_bins = bins.reshape(-1)
            # advantages are already per-goal standardized; report the PRE-norm
            # raw advantage mean per goal instead (the sharpest collapse signal).
            raw_adv = (returns - values).reshape(-1)
            for b in range(NUM_BINS):
                m = flat_bins == b
                if m.any():
                    self.track_data(f"PerGoal / raw_advantage_mean_bin{b}", raw_adv[m].mean().item())
                    self.track_data(
                        f"PerGoal / value_scale_bin{b}",
                        float(self._pg_value_pre[b].running_variance.sqrt().mean().item()),
                    )

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []

            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in self.memory.sample(
                names=self._tensors_names, batch_size=len(self.memory), mini_batches=self.cfg.mini_batches
            ):

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                        "states": self._state_preprocessor(sampled_states, train=not epoch),
                    }

                    _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if self.cfg.kl_threshold and kl_divergence > self.cfg.kl_threshold:
                        break

                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _ = self.value.act(inputs, role="value")
                    if self.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()),
                            self.cfg.grad_norm_clip,
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        self.track_data(
            "Loss / Policy loss", cumulative_policy_loss / (self.cfg.learning_epochs * self.cfg.mini_batches)
        )
        self.track_data(
            "Loss / Value loss", cumulative_value_loss / (self.cfg.learning_epochs * self.cfg.mini_batches)
        )
        if self.cfg.entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self.cfg.learning_epochs * self.cfg.mini_batches)
            )
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

    # ----------------------------------------------------- per-goal GAE helper
    @staticmethod
    def _compute_gae_per_goal(
        *,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        values: torch.Tensor,
        last_values: torch.Tensor,
        bins: torch.Tensor,
        discount_factor: float = 0.99,
        lambda_coefficient: float = 0.95,
        time_limit_bootstrap: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE with **per-goal** advantage standardization.

        Identical to skrl ``compute_gae`` except the final normalization is done
        within each commanded-bin group rather than over the whole batch.
        """
        advantage = 0
        advantages = torch.zeros_like(rewards)
        not_done = ((terminated | truncated) if time_limit_bootstrap else terminated).logical_not()
        memory_size = rewards.shape[0]

        for i in reversed(range(memory_size)):
            next_values = values[i + 1] if i < memory_size - 1 else last_values
            advantage = (
                rewards[i] - values[i]
                + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
            )
            advantages[i] = advantage

        returns = advantages + values

        # per-goal advantage normalization
        flat_bins = bins.reshape(-1)
        flat_adv = advantages.reshape(-1)
        normed = torch.empty_like(flat_adv)
        for b in range(NUM_BINS):
            mask = flat_bins == b
            if mask.any():
                grp = flat_adv[mask]
                normed[mask] = (grp - grp.mean()) / (grp.std() + 1e-8)
        advantages = normed.reshape_as(advantages)

        return returns, advantages


__all__ = ["PerGoalPPO", "NUM_BINS"]
