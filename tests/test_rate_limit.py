from __future__ import annotations

import math
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from rag_system.rate_limit import TokenBucketRateLimiter


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class TokenBucketRateLimiterTests(unittest.TestCase):
    def test_capacity_blocks_bursts_and_returns_retry_after(self) -> None:
        clock = FakeClock(100.0)
        limiter = TokenBucketRateLimiter(
            rate_per_second=2.0,
            capacity=2.0,
            clock=clock,
        )

        first = limiter.acquire("tenant")
        second = limiter.acquire("tenant")
        denied = limiter.acquire("tenant")

        self.assertTrue(first.allowed)
        self.assertTrue(second.allowed)
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.remaining_tokens, 0.0)
        self.assertAlmostEqual(denied.retry_after_seconds, 0.5)

        clock.advance(0.5)
        recovered = limiter.acquire("tenant")
        self.assertTrue(recovered.allowed)
        self.assertEqual(recovered.retry_after_seconds, 0.0)

    def test_tenants_are_isolated_and_clock_rollback_never_refills(self) -> None:
        clock = FakeClock(10.0)
        limiter = TokenBucketRateLimiter(
            rate_per_second=1.0,
            capacity=1.0,
            clock=clock,
        )
        self.assertTrue(limiter.acquire("first").allowed)
        self.assertTrue(limiter.acquire("second").allowed)

        clock.value = 5.0
        rolled_back = limiter.acquire("first")
        self.assertFalse(rolled_back.allowed)
        self.assertAlmostEqual(rolled_back.retry_after_seconds, 1.0)

        clock.value = 10.5
        half_refill = limiter.acquire("first")
        self.assertFalse(half_refill.allowed)
        self.assertAlmostEqual(half_refill.remaining_tokens, 0.5)

    def test_ttl_and_lru_keep_tenant_state_bounded(self) -> None:
        clock = FakeClock()
        limiter = TokenBucketRateLimiter(
            rate_per_second=1.0,
            capacity=1.0,
            max_keys=2,
            key_ttl_seconds=5.0,
            clock=clock,
        )
        limiter.acquire("first")
        limiter.acquire("second")
        clock.advance(0.1)
        limiter.acquire("first")
        limiter.acquire("third")
        self.assertEqual(limiter.key_count, 2)

        reintroduced = limiter.acquire("second")
        self.assertTrue(reintroduced.allowed)
        self.assertEqual(limiter.key_count, 2)

        clock.advance(5.0)
        self.assertEqual(limiter.prune(), 2)
        self.assertEqual(limiter.key_count, 0)
        self.assertTrue(limiter.acquire("first").allowed)

    def test_invalid_negative_non_finite_and_impossible_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TokenBucketRateLimiter(rate_per_second=0, capacity=1)
        with self.assertRaises(ValueError):
            TokenBucketRateLimiter(rate_per_second=1, capacity=-1)
        with self.assertRaises(ValueError):
            TokenBucketRateLimiter(rate_per_second=math.inf, capacity=1)

        limiter = TokenBucketRateLimiter(rate_per_second=1, capacity=2)
        for value in (False, 0.0, -1.0, math.nan, math.inf, 3.0):
            with self.subTest(tokens=value):
                with self.assertRaises(ValueError):
                    limiter.acquire("tenant", tokens=value)

    def test_non_finite_clock_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            TokenBucketRateLimiter(
                rate_per_second=1,
                capacity=1,
                clock=lambda: math.nan,
            )

    def test_concurrent_burst_never_exceeds_capacity(self) -> None:
        capacity = 10
        limiter = TokenBucketRateLimiter(
            rate_per_second=1.0,
            capacity=capacity,
            clock=lambda: 0.0,
        )
        barrier = threading.Barrier(40)

        def attempt(_: int) -> bool:
            barrier.wait()
            return limiter.acquire("shared-tenant").allowed

        with ThreadPoolExecutor(max_workers=40) as executor:
            allowed = list(executor.map(attempt, range(40)))

        self.assertEqual(sum(allowed), capacity)
        self.assertEqual(limiter.key_count, 1)


if __name__ == "__main__":
    unittest.main()
