from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def atomic_write_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, destination)


def update_latest_symlink(target: str | Path, link: str | Path) -> None:
    """Point ``link`` at ``target`` atomically, using a relative target.

    A relative target survives moving the checkpoint directory as a whole, and
    the tmp-then-replace dance means a crash can never leave the link missing
    or dangling between unlink and symlink.
    """
    target_path = Path(target).resolve()
    link_path = Path(link)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_dir() and not link_path.is_symlink():
        shutil.rmtree(link_path)
    # Resolve both sides so the ".." count stays correct when the output tree
    # itself sits behind a symlinked mount.
    relative_target = os.path.relpath(target_path, link_path.parent.resolve())
    temporary = link_path.parent / (link_path.name + ".tmp")
    if temporary.is_symlink() or temporary.is_file():
        temporary.unlink()
    temporary.symlink_to(relative_target, target_is_directory=True)
    os.replace(temporary, link_path)
