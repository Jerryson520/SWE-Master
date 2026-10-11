"""rLLM adapter for the Leaf environment owned by R2E-Gym."""

from r2egym.agenthub.leaf.environment import LeafRepoEnv
from rllm.environments.base.base_env import BaseEnv


class SWELeafEnv(LeafRepoEnv, BaseEnv):
    """Expose the R2E Leaf runtime through rLLM's environment interface."""
