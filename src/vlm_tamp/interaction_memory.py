"""Bounded, task-independent visual transitions for a structured robot actor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InteractionMemory:
    """Retain complete executed transitions; never turn a hypothesis into a fact."""

    mode: str
    """Either recent-window or anchored retention."""
    slots: int = 8
    """Maximum retained transitions, including the latest."""
    first_control: int = 60
    """Measured initial control count after the shared reset."""
    last_control: int = 600
    """Existing episode control limit; anchors do not impose action boundaries."""
    records: list[dict[str, Any]] = field(default_factory=list)
    """Small transition records, with image paths rather than embedded pixels."""
    anchors: dict[int, int] = field(default_factory=dict)
    """Temporal anchor number to immutable transition index."""
    pending: list[dict[str, Any]] = field(default_factory=list)
    """Immediate non-motion feedback, distinct from executed transitions."""

    def __post_init__(self) -> None:
        """Validate a declared memory condition without substituting another."""
        if self.mode not in {'recent', 'anchored'} or self.slots < 2:
            raise ValueError('Use recent or anchored memory with at least two slots')

    def add(self, record: dict[str, Any]) -> bool:
        """Commit positive-length execution; keep non-motion feedback in the tail."""
        if record['after_control'] <= record['before_control']:
            self.pending.append(record)
            self.pending = self.pending[-self.slots:]
            return False
        self.pending = []
        index = len(self.records)
        self.records.append(record)
        duration = self.last_control-self.first_control
        for anchor in range(self.slots-1):
            target = self.first_control+anchor*duration/self.slots
            if anchor not in self.anchors and record['after_control'] > target:
                self.anchors[anchor] = index
        return True

    def selected(self) -> list[dict[str, Any]]:
        """Return chronological, deduplicated records under the chosen condition."""
        if self.mode == 'recent':
            return (self.records if len(self.records)<=self.slots else
                    [self.records[0],*self.records[-(self.slots-1):]])
        indices = sorted(set(self.anchors.values()) | ({len(self.records)-1} if self.records else set()))
        return [self.records[index] for index in indices]

    def context(self) -> dict[str, Any]:
        """Mark omitted intervals explicitly instead of implying adjacent frames."""
        events: list[dict[str, Any]] = []
        previous = self.first_control
        for record in self.selected():
            if record['before_control'] > previous:
                events.append({'type': 'TRAJECTORY_GAP', 'after_control': previous,
                               'before_control': record['before_control']})
            events.append({'type': 'executed_transition', **record})
            previous = record['after_control']
        return {'mode': self.mode, 'slot_limit': self.slots, 'events': events,
                'recent_nonmotion_feedback': self.pending,
                'total_executed_transitions': len(self.records),
                'retained_executed_transitions': len(self.selected()),
                'source': 'this episode only; no demonstrations or inherited task memories'}
