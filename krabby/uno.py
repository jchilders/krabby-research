"""krabby uno — run the gamepad control-loop client inside the locomotion container."""
from __future__ import annotations

import subprocess
import sys
from typing import Optional

from krabby._docker import uno_cmd
from krabby._state import resolve_image_ref


def cmd_uno(image_ref: Optional[str] = None, extra_args: Optional[list[str]] = None) -> None:
    ref = resolve_image_ref(image_ref)
    cmd = uno_cmd(ref, extra_args or [])
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
