"""Launch robot_keyframe_kit's keyframe-editor for Microban on port 8081.

The editor always opens viser's default port 8080, which `microban-play
--viewer viser` uses too; this wrapper moves it to EDITOR_PORT (override with
the KEYFRAME_EDITOR_PORT environment variable) and fills in the Microban scene,
config and initial get-up keyframes. Any other keyframe-editor option can be
appended, e.g. `--data keyframes/getup.lz4` to continue from a saved file.

  cd ~/mero/keyframe_editor
  uv run python run_editor.py                 # starts from init/getup_init.lz4
  uv run python run_editor.py --data keyframes/getup.lz4
"""

import functools
import os
import sys
from pathlib import Path

import viser

HERE = Path(__file__).resolve().parent
EDITOR_PORT = int(os.environ.get("KEYFRAME_EDITOR_PORT", "8081"))

_init = viser.ViserServer.__init__


@functools.wraps(_init)
def _init_on_editor_port(self, host="0.0.0.0", port=8080, *args, **kwargs):
  _init(self, host, EDITOR_PORT if port == 8080 else port, *args, **kwargs)


viser.ViserServer.__init__ = _init_on_editor_port

from robot_keyframe_kit.editor import main  # noqa: E402

if __name__ == "__main__":
  args = sys.argv[1:]
  if "--data" not in args:
    args += ["--data", str(HERE / "init" / "getup_init.lz4")]
  sys.argv = [sys.argv[0], str(HERE / "microban_scene.xml"),
              "--config", str(HERE / "config.yaml"), *args]
  main()
