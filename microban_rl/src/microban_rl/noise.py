"""Temporally correlated exploration noise for rsl-rl PPO.

rsl-rl's GaussianDistribution draws independent noise every policy step. At
50 Hz with a ~0.1 s position-servo response, consecutive independent kicks
mostly cancel out and the motors barely move. ColoredGaussianDistribution keeps
the same Gaussian policy (mean, learned std) but samples the rollout noise from
a per-environment AR(1) / Ornstein-Uhlenbeck process:

  eps_t = rho * eps_{t-1} + sqrt(1 - rho^2) * xi_t,   xi_t ~ N(0, I)

eps_t has a stationary N(0, I) marginal, so each action is still distributed as
N(mean, std^2) and PPO's log-probabilities / KL / entropy are computed exactly as
before from that marginal. What changes is only that consecutive noise samples
are correlated with time constant `correlation_time_s`. (PPO still treats the
per-step importance ratios as independent; that is the usual approximation for
colored-noise exploration in on-policy RL.)

Use in a runner cfg:
  distribution_cfg={"class_name": "microban_rl.noise:ColoredGaussianDistribution",
                    "init_std": 1.0, "std_type": "scalar",
                    "correlation_time_s": 0.25, "step_dt": 0.02}
"""

from __future__ import annotations

import math

import torch
from rsl_rl.modules.distribution import GaussianDistribution


class ColoredGaussianDistribution(GaussianDistribution):
  def __init__(
    self,
    output_dim: int,
    init_std: float = 1.0,
    std_range: tuple[float, float] = (1e-6, 1e6),
    std_type: str = "scalar",
    learn_std: bool = True,
    correlation_time_s: float = 0.25,
    step_dt: float = 0.02,
  ) -> None:
    super().__init__(output_dim, init_std, std_range, std_type, learn_std)
    self.rho = math.exp(-step_dt / correlation_time_s)
    self._eps: torch.Tensor | None = None

  def sample(self) -> torch.Tensor:
    mean, std = self.mean, self.std
    if self._eps is None:
      # The first sampled batch is the rollout batch ([num_envs, A]).
      self._eps = torch.randn_like(mean)
    elif self._eps.shape == mean.shape:
      self._eps = self.rho * self._eps + math.sqrt(1.0 - self.rho**2) * torch.randn_like(
        mean
      )
    else:
      # Other batch shapes (PPO mini-batches) get independent noise and leave the
      # rollout noise process untouched.
      return mean + std * torch.randn_like(mean)
    return mean + std * self._eps
