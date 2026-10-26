"""Per-episode pid lock files so several worker processes can share one queue."""
from __future__ import annotations

import os
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def try_lock(directory: Path, name: str) -> bool:
    lock = directory / name
    if lock.exists():
        try:
            if _pid_alive(int(lock.read_text().strip())):
                return False
        except ValueError:
            pass
    directory.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    return True


def unlock(directory: Path, name: str) -> None:
    (directory / name).unlink(missing_ok=True)
