"""Discover sessions under data_sensor/row and assign binary / multiclass labels.

Binary: ``AdultManFall`` → label 1; all other folders → label 0.

Multiclass (for research / disambiguation):

- ``0`` — everyday ADL (all folders except the two below)
- ``1`` — person fall (``AdultManFall``)
- ``2`` — phone-only drop / device impact (``IndoorPhoneFall``), still ``label=0`` in binary mode
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

FALL_ACTIVITY = "AdultManFall"
PHONE_DROP_ACTIVITY = "IndoorPhoneFall"

MULTICLASS_ADL = 0
MULTICLASS_PERSON_FALL = 1
MULTICLASS_PHONE_DROP = 2


def multiclass_for_activity(activity: str) -> int:
    if activity == FALL_ACTIVITY:
        return MULTICLASS_PERSON_FALL
    if activity == PHONE_DROP_ACTIVITY:
        return MULTICLASS_PHONE_DROP
    return MULTICLASS_ADL


@dataclass(frozen=True)
class RowSession:
    """One recording folder (e.g. .../IndoorAdultManWalking/01_0427_ByLiu)."""

    activity: str
    session_id: str
    path: Path
    label: int
    multiclass: int

    @property
    def group_key(self) -> str:
        return f"{self.activity}/{self.session_id}"


def iter_row_sessions(row_root: Path) -> Iterator[RowSession]:
    if not row_root.is_dir():
        raise FileNotFoundError(row_root)
    for activity_dir in sorted(row_root.iterdir()):
        if not activity_dir.is_dir():
            continue
        activity = activity_dir.name
        label = 1 if activity == FALL_ACTIVITY else 0
        mc = multiclass_for_activity(activity)
        for session_dir in sorted(activity_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            yield RowSession(
                activity=activity,
                session_id=session_dir.name,
                path=session_dir,
                label=label,
                multiclass=mc,
            )


def list_row_sessions(row_root: Path) -> List[RowSession]:
    return list(iter_row_sessions(row_root))
