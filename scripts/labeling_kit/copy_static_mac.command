#!/bin/bash
# Double-click after the first frame of a video is labeled and checked. Close X-AnyLabeling first.
cd "$(dirname "$0")" || exit 1
VENV="${XAL_VENV:-$HOME/xal}"
"$VENV/bin/python" propagate_static.py
read -r -p "Готово. Нажмите Enter, чтобы закрыть"
