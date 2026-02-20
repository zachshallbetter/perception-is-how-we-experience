"""Lightweight metrics collector for service runs."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class MetricsCollector:
    """In-memory counters and observations with JSON snapshot output."""

    counters: Dict[str, int] = field(default_factory=dict)
    observations: Dict[str, List[float]] = field(default_factory=dict)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def observe(self, name: str, value: float) -> None:
        if name not in self.observations:
            self.observations[name] = []
        self.observations[name].append(float(value))

    def snapshot(self) -> Dict:
        obs_summary = {}
        for name, values in self.observations.items():
            if not values:
                continue
            obs_summary[name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
        return {
            "counters": dict(sorted(self.counters.items())),
            "observations": dict(sorted(obs_summary.items())),
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            json.dump(self.snapshot(), file, indent=2, sort_keys=True)
