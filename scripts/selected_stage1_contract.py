"""Resolve only the checkpoint selected by a completed Stage1 chain gate."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_selected_stage1(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    selection = json.loads((checkpoint.parent / "selected_stage1.json").read_text())
    assert selection["status"] == "PASS"
    assert Path(selection["checkpoint"]).resolve() == checkpoint
    step = int(selection["global_step"])
    assert step == 6283 or (0 < step <= 6000 and step % 500 == 0)
    expected_epoch = 1 if step == 6283 else 0
    assert int(selection["epoch"]) == expected_epoch
    for filename, digest in selection["checkpoint_hashes"].items():
        assert sha256(checkpoint / filename) == digest, ("selected checkpoint changed", filename)
    return selection
