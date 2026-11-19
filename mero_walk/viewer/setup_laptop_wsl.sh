#!/usr/bin/env bash
# 노트북(WSL2 Ubuntu 22.04/24.04 또는 Ubuntu)에서 한 번: Gazebo Harmonic + 뷰어 파이썬 환경
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
sudo apt-get update
sudo apt-get install -y curl lsb-release gnupg python3-venv
if ! command -v gz >/dev/null; then
  sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y gz-harmonic
fi
sudo apt-get install -y python3-gz-transport13 python3-gz-msgs10 || true
# gz 파이썬 바인딩(시스템)을 그대로 쓰면서 mujoco만 추가
python3 -m venv --system-site-packages "$HERE/.venv"
"$HERE/.venv/bin/pip" install -q mujoco numpy
"$HERE/.venv/bin/python" -c "from gz.transport13 import Node; import mujoco; print('OK: gz-transport13 + mujoco', mujoco.__version__)"
echo "실행: $HERE/.venv/bin/python $HERE/mero_viewer.py --robot <로봇IP>"
