#!/usr/bin/env bash
source "${SCRIPT_DIR:?}/config.sh"
[[ -d "$CHD_REPO/nnunetv2" && -x "$NNUNET_ENV/bin/python" ]] || { echo 'Check CHD_REPO and NNUNET_ENV in config.sh' >&2; exit 1; }
export PATH="$NNUNET_ENV/bin:$PATH"
export PYTHONPATH="$CHD_REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$CHD_REPO"
python - <<'PY'
import os
import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
cls=recursive_find_python_class(os.path.join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer'), os.environ['TRAINER'], 'nnunetv2.training.nnUNetTrainer')
assert cls is not None, f"Trainer {os.environ['TRAINER']} unavailable; select an installed trainer in config.sh"
print('nnU-Net:', nnunetv2.__path__[0], 'trainer:', cls.__name__)
PY

if [[ -n "${RUN_DIR:-}" ]]; then
  python "$SCRIPT_DIR/record_job.py"
fi
