"""Shared loading for step-level classification manifests.

Every dataset that consumes ``ClassificationRecord`` JSONL manifests goes
through :func:`load_step_records`, so format validation and error reporting
live in exactly one place.  (``CachedFrameDataset`` reads the offline cache's
own ``metadata.jsonl`` schema and is intentionally not covered here.)
"""

from __future__ import annotations

import json
from pathlib import Path

from aerial_openvla.data.schema import ClassificationRecord


def load_step_records(manifest_path: str | Path) -> list[ClassificationRecord]:
    """Parse a step-level JSONL manifest into validated records.

    Raises ``ValueError`` with the offending line number on malformed rows and
    on empty manifests, matching the behaviour the datasets always had.
    """
    path = Path(manifest_path)
    records: list[ClassificationRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(ClassificationRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}") from exc
    if not records:
        raise ValueError(f"Manifest contains no records: {path}")
    return records
