#!/bin/bash
# Double-click to label. First run installs X-AnyLabeling 4.1.3 into ~/xal (a few minutes).
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$PATH"
VENV="${XAL_VENV:-$HOME/xal}"

if [ ! -x "$VENV/bin/xanylabeling" ]; then
  PY=""
  for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
      PY="$candidate"
      break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Нужен Python 3.11 или новее: https://www.python.org/downloads/ — установите и запустите снова."
    read -r -p "Нажмите Enter, чтобы закрыть"
    exit 1
  fi
  echo "Первый запуск: устанавливаю X-AnyLabeling (несколько минут)..."
  if ! "$PY" -m venv "$VENV" || ! "$VENV/bin/pip" install -q x-anylabeling-cvhub==4.1.3; then
    echo "Установка не удалась. Пришлите Диару текст выше (без скриншотов кадров)."
    read -r -p "Нажмите Enter, чтобы закрыть"
    exit 1
  fi
fi

"$VENV/bin/xanylabeling" --filename images --labels classes.txt --validatelabel exact --autosave --nodata --no-auto-update-check
