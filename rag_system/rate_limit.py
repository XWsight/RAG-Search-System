"""Thread-safe, bounded per-tenant token-bucket rate limiting."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining_tokens: float
    retry_after_seconds: float
    capacity: float
    rate_per_second: float


@dataclass(slots=True)
class _Bucket:
    tokens: float
    last_refill: float
    last_seen: float


class TokenBucketRateLimiter:
    """A per-process limiter with bounded tenant state and deterministic clocks."""

    def __init__(
        self,
        *,
        rate_per_second: float,
        capacity: float,
        max_keys: int = 10_000,
        key_ttl_seconds: float = 3_600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(rate_per_second) or rate_per_second <= 0:
            raise ValueError("rate_per_second must be finite and positive")
        if not math.isfinite(capacity) or capacity <= 0:
            raise ValueError("capacity must be finite and positive")
        if not isinstance(max_keys, int) or isinstance(max_keys, bool) or max_keys < 1:
            raise ValueError("max_keys must be a positive integer")
        if not math.isfinite(key_ttl_seconds) or key_ttl_seconds <= 0:
            raise ValueError("key_ttl_seconds must be finite and positive")

        self.rate_per_second = float(rate_per_second)
        self.capacity = float(capacity)
        self.max_keys = max_keys
        self.key_ttl_seconds = float(key_ttl_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: OrderedDict[bytes, _Bucket] = OrderedDict()
        self._last_clock = self._raw_clock()

    def acquire(self, tenant_key: str, *, tokens: float = 1.0) -> RateLimitDecision:
        key = self._key_digest(tenant_key)
        if isinstance(tokens, bool):
            raise ValueError("tokens must be finite and positive")
        requested = float(tokens)
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError("tokens must be finite and positive")
        if requested > self.capacity:
            raise ValueError("tokens cannot exceed bucket capacity")

        with self._lock:
            now = self._safe_now()
            self._prune_expired(now)
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self.max_keys:
                    self._buckets.popitem(last=False)
                bucket = _Bucket(tokens=self.capacity, last_refill=now, last_seen=now)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, now - bucket.last_refill)
                bucket.tokens = min(
                    self.capacity,
                    bucket.tokens + elapsed * self.rate_per_second,
                )
                bucket.last_refill = now
                bucket.last_seen = now
                self._buckets.move_to_end(key)

            tolerance = max(1e-12, self.capacity * 1e-12)
            if bucket.tokens + tolerance >= requested:
                bucket.tokens = max(0.0, bucket.tokens - requested)
                return RateLimitDecision(
                    allowed=True,
                    remaining_tokens=bucket.tokens,
                    retry_after_seconds=0.0,
                    capacity=self.capacity,
                    rate_per_second=self.rate_per_second,
                )

            deficit = max(0.0, requested - bucket.tokens)
            return RateLimitDecision(
                allowed=False,
                remaining_tokens=max(0.0, bucket.tokens),
                retry_after_seconds=deficit / self.rate_per_second,
                capacity=self.capacity,
                rate_per_second=self.rate_per_second,
            )

    def prune(self) -> int:
        """Remove expired tenant buckets and return the number removed."""

        with self._lock:
            return self._prune_expired(self._safe_now())

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()

    @property
    def key_count(self) -> int:
        with self._lock:
            return len(self._buckets)

    def _safe_now(self) -> float:
        observed = self._raw_clock()
        if observed < self._last_clock:
            return self._last_clock
        self._last_clock = observed
        return observed

    def _raw_clock(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise RuntimeError("clock returned a non-finite value")
        return value

    def _prune_expired(self, now: float) -> int:
        removed = 0
        while self._buckets:
            _, bucket = next(iter(self._buckets.items()))
            if now - bucket.last_seen < self.key_ttl_seconds:
                break
            self._buckets.popitem(last=False)
            removed += 1
        return removed

    @staticmethod
    def _key_digest(tenant_key: str) -> bytes:
        if not isinstance(tenant_key, str):
            raise TypeError("tenant_key must be a string")
        normalized = tenant_key.strip()
        if not normalized:
            raise ValueError("tenant_key cannot be empty")
        if len(normalized) > 256:
            raise ValueError("tenant_key is too long")
        return hashlib.blake2s(normalized.encode("utf-8"), digest_size=16).digest()


__all__ = ["RateLimitDecision", "TokenBucketRateLimiter"]
