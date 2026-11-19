#!/usr/bin/env bash
# MERO 로봇(오렌지파이)에서 한 번 실행: 권한·패키지 확인. 배터리 감시 서비스는 --battery-service.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${MERO_PY:-/home/orangepi/microban/.venv/bin/python}"
USER_NAME="${SUDO_USER:-$(id -un)}"

echo "== Python: $PY"
"$PY" -c "import sys; print(sys.version)"
missing=$("$PY" - <<'PYEOF'
import importlib.util
need = {"numpy": "numpy", "onnxruntime": "onnxruntime", "serial": "pyserial", "smbus2": "smbus2"}
print(" ".join(p for m, p in need.items() if importlib.util.find_spec(m) is None))
PYEOF
)
if [ -n "$missing" ]; then
  echo "== 설치: $missing"
  "$PY" -m pip install $missing
else
  echo "== 필요한 패키지 모두 있음 (numpy, onnxruntime, pyserial, smbus2)"
fi

echo "== 장치 확인"
ls -l /dev/ttyAS0 /dev/i2c-1 || true
for g in dialout i2c; do
  if getent group "$g" >/dev/null && ! id -nG "$USER_NAME" | grep -qw "$g"; then
    echo "   $USER_NAME 를 $g 그룹에 추가 (sudo 필요, 다시 로그인해야 적용)"
    sudo usermod -aG "$g" "$USER_NAME"
  fi
done
if systemctl is-active --quiet serial-getty@ttyAS0.service; then
  echo "   경고: ttyAS0에 로그인 콘솔이 붙어 있습니다. sudo systemctl disable --now serial-getty@ttyAS0"
fi

if [ "${1:-}" = "--battery-service" ]; then
  sudo cp "$HERE/systemd/mero-battery.service" /etc/systemd/system/
  sudo sed -i "s#/home/orangepi/mero_walk#$HERE#; s#/home/orangepi/microban/.venv/bin/python#$PY#" /etc/systemd/system/mero-battery.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now mero-battery.service
  systemctl --no-pager status mero-battery.service | head -5
fi
echo "== 완료. 다음: cd $HERE && $PY -m mero check"
