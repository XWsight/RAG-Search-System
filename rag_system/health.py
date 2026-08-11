"""Fail-closed, side-effect-free readiness probe composition."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass


_PROBE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


@dataclass(frozen=True, slots=True)
class HealthProbe:
    name: str
    check: Callable[[], bool]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _PROBE_NAME.fullmatch(self.name) is None:
            raise ValueError("probe name is invalid")
        if not callable(self.check):
            raise TypeError("probe check must be callable")


@dataclass(frozen=True, slots=True)
class ProbeResult:
    name: str
    ready: bool


class ReadinessMonitor:
    """Evaluate a fixed set of local dependency probes without leaking errors."""

    def __init__(self, probes: Sequence[HealthProbe]) -> None:
        normalized = tuple(probes)
        if not normalized:
            raise ValueError("at least one readiness probe is required")
        if any(not isinstance(item, HealthProbe) for item in normalized):
            raise TypeError("probes must contain HealthProbe values")
        names = tuple(item.name for item in normalized)
        if len(names) != len(set(names)):
            raise ValueError("readiness probe names must be unique")
        self._probes = normalized

    def ready(self) -> bool:
        return all(item.ready for item in self.snapshot())

    def snapshot(self) -> tuple[ProbeResult, ...]:
        results: list[ProbeResult] = []
        for probe in self._probes:
            try:
                ready = probe.check() is True
            except Exception:
                ready = False
            results.append(ProbeResult(probe.name, ready))
        return tuple(results)


__all__ = ["HealthProbe", "ProbeResult", "ReadinessMonitor"]
