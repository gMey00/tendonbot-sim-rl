"""Custom learning components for shirt_distribute (goal-conditioned PPO).

See ``doc/reports/RESEARCH_REPORT_goal_conditioned_mode_collapse`` — these
implement recommendation #1 (per-goal value/advantage normalization + per-goal
value heads) to break the per-seed 2-of-3 goal mode collapse.
"""

from .goal_value import GoalMultiHeadValue, goal_multi_head_value_factory
from .per_goal_ppo import NUM_BINS, PerGoalPPO
from .runner import PerGoalRunner

__all__ = [
    "GoalMultiHeadValue",
    "goal_multi_head_value_factory",
    "NUM_BINS",
    "PerGoalPPO",
    "PerGoalRunner",
]
