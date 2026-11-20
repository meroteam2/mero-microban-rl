"""Microban tasks for MJLab.

Importing this package registers the ``Microban-Stand``, ``Microban-Crawl``,
``Microban-Crawl-CPG``, ``Microban-Crawl-Arms``, ``Microban-Crawl-Arms-2``, ``Microban-GetUp``, ``Microban-GetUp-2``, ``Microban-GetUp-3``, ``Microban-GetUp-4``, ``Microban-GetUp-5``, ``Microban-Walk``, ``Microban-Walk-DR`` and ``Microban-Walk-DR2``
tasks with MJLab.
"""

import os

# WSL2 has no usable EGL, but offscreen rendering works through WSLg's GLFW.
# Must be set before mjlab/mujoco are imported (mjlab defaults it to egl).
if "microsoft" in os.uname().release.lower() and os.environ.get("DISPLAY"):
  os.environ.setdefault("MUJOCO_GL", "glfw")

from mjlab.tasks.registry import register_mjlab_task  # noqa: E402

from microban_rl.crawl_arms_env_cfg import (  # noqa: E402
  ArmNoiseRunner,
  microban_crawl_arms2_env_cfg,
  microban_crawl_arms2_ppo_runner_cfg,
  microban_crawl_arms_env_cfg,
  microban_crawl_arms_ppo_runner_cfg,
)
from microban_rl.crawl_cpg_env_cfg import (  # noqa: E402
  microban_crawl_cpg_env_cfg,
  microban_crawl_cpg_ppo_runner_cfg,
)
from microban_rl.crawl_env_cfg import (  # noqa: E402
  microban_crawl_env_cfg,
  microban_crawl_ppo_runner_cfg,
)
from microban_rl.stand_env_cfg import (  # noqa: E402
  microban_stand_env_cfg,
  microban_stand_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Microban-Stand",
  env_cfg=microban_stand_env_cfg(),
  play_env_cfg=microban_stand_env_cfg(play=True),
  rl_cfg=microban_stand_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Microban-Crawl",
  env_cfg=microban_crawl_env_cfg(),
  play_env_cfg=microban_crawl_env_cfg(play=True),
  rl_cfg=microban_crawl_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Microban-Crawl-CPG",
  env_cfg=microban_crawl_cpg_env_cfg(),
  play_env_cfg=microban_crawl_cpg_env_cfg(play=True),
  rl_cfg=microban_crawl_cpg_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Microban-Crawl-Arms",
  env_cfg=microban_crawl_arms_env_cfg(),
  play_env_cfg=microban_crawl_arms_env_cfg(play=True),
  rl_cfg=microban_crawl_arms_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Microban-Crawl-Arms-2",
  env_cfg=microban_crawl_arms2_env_cfg(),
  play_env_cfg=microban_crawl_arms2_env_cfg(play=True),
  rl_cfg=microban_crawl_arms2_ppo_runner_cfg(),
  runner_cls=ArmNoiseRunner,
)

from microban_rl.getup_env_cfg import (  # noqa: E402
  microban_getup_env_cfg,
  microban_getup_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Microban-GetUp",
  env_cfg=microban_getup_env_cfg(),
  play_env_cfg=microban_getup_env_cfg(play=True),
  rl_cfg=microban_getup_ppo_runner_cfg(),
)

from microban_rl.getup2_env_cfg import (  # noqa: E402
  microban_getup2_env_cfg,
  microban_getup2_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Microban-GetUp-2",
  env_cfg=microban_getup2_env_cfg(),
  play_env_cfg=microban_getup2_env_cfg(play=True),
  rl_cfg=microban_getup2_ppo_runner_cfg(),
)

from microban_rl.getup3_env_cfg import (  # noqa: E402
  microban_getup3_env_cfg,
  microban_getup3_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Microban-GetUp-3",
  env_cfg=microban_getup3_env_cfg(),
  play_env_cfg=microban_getup3_env_cfg(play=True),
  rl_cfg=microban_getup3_ppo_runner_cfg(),
)

from microban_rl.getup4_env_cfg import (  # noqa: E402
  microban_getup4_env_cfg,
  microban_getup4_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Microban-GetUp-4",
  env_cfg=microban_getup4_env_cfg(),
  play_env_cfg=microban_getup4_env_cfg(play=True),
  rl_cfg=microban_getup4_ppo_runner_cfg(),
)

from microban_rl.getup5_env_cfg import (  # noqa: E402
  microban_getup5_env_cfg,
  microban_getup5_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Microban-GetUp-5",
  env_cfg=microban_getup5_env_cfg(),
  play_env_cfg=microban_getup5_env_cfg(play=True),
  rl_cfg=microban_getup5_ppo_runner_cfg(),
)

from microban_rl.walk_env_cfg import (  # noqa: E402
  microban_walk_dr2_env_cfg,
  microban_walk_dr2_ppo_runner_cfg,
  microban_walk_dr_env_cfg,
  microban_walk_dr_ppo_runner_cfg,
  microban_walk_env_cfg,
  microban_walk_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Microban-Walk",
  env_cfg=microban_walk_env_cfg(),
  play_env_cfg=microban_walk_env_cfg(play=True),
  rl_cfg=microban_walk_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Microban-Walk-DR",
  env_cfg=microban_walk_dr_env_cfg(),
  play_env_cfg=microban_walk_dr_env_cfg(play=True),
  rl_cfg=microban_walk_dr_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Microban-Walk-DR2",
  env_cfg=microban_walk_dr2_env_cfg(),
  play_env_cfg=microban_walk_dr2_env_cfg(play=True),
  rl_cfg=microban_walk_dr2_ppo_runner_cfg(),
)
