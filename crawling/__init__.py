"""Crawling utilities and rate limiting."""

from __future__ import annotations

from typing import Dict


class RateLimiter:
    """Simple exponential backoff rate limiter."""

    def __init__(self, min_delay: float, max_delay: float | None = None):
        if max_delay is None:
            max_delay = min_delay
        self.base_min = min_delay
        self.base_max = max_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.consecutive_failures = 0

    def _sample_delay(self) -> float:
        import random

        return random.uniform(self.min_delay, self.max_delay)

    def wait(self) -> None:
        import time

        time.sleep(self._sample_delay())

    def record_error(self) -> None:
        self.consecutive_failures += 1
        self.min_delay = min(self.max_delay, self.min_delay * 2)
        self.max_delay = min(self.max_delay, self.max_delay * 2)

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.min_delay = self.base_min
        self.max_delay = self.base_max


class DynamicRateLimiter:
    """Manage per-host rate limiters with exponential backoff."""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0) -> None:
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.limiters: Dict[str, RateLimiter] = {}

    def get(self, host: str) -> RateLimiter:
        if host not in self.limiters:
            self.limiters[host] = RateLimiter(self.base_delay, self.max_delay)
        return self.limiters[host]


rate_limiter = DynamicRateLimiter()

__all__ = ["rate_limiter", "DynamicRateLimiter"]
