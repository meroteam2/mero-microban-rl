"""Microban tasks for MJLab.

Importing this package registers the ``Microban-Stand``
tasks with MJLab.
"""

import os

# WSL2 has no usable EGL, but offscreen rendering works through WSLg's GLFW.
# Must be set before mjlab/mujoco are imported (mjlab defaults it to egl).
if "microsoft" in os.uname().release.lower() and os.environ.get("DISPLAY"):
  os.environ.setdefault("MUJOCO_GL", "glfw")

from mjlab.tasks.registry import register_mjlab_task  # noqa: E402

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
