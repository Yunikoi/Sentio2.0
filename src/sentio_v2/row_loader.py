"""Discover sessions under data_sensor/row and assign binary labels.

Only recordings under activity folder ``AdultManFall`` are treated as falls (label 1).
All other activity folders are non-falls (label 0), including e.g. ``IndoorPhoneFall``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

FALL_ACTIVITY = "AdultManFall"


@dataclass(frozen=True)
class RowSession:
    """One recording folder (e.g. .../IndoorAdultManWalking/01_0427_ByLiu)."""

    activity: str
    session_id: str
    path: Path
    label: int

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
        for session_dir in sorted(activity_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            yield RowSession(
                activity=activity,
                session_id=session_dir.name,
                path=session_dir,
                label=label,
            )


def list_row_sessions(row_root: Path) -> List[RowSession]:
    return list(iter_row_sessions(row_root))
