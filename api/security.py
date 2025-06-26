from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware


auth_scheme = HTTPBearer(auto_error=False)


def require_token(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> None:
    """Validate bearer token if ``API_TOKEN`` is set."""
    token = os.environ.get("API_TOKEN")
    if token and (credentials is None or credentials.credentials != token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Naïve per-IP rate limiter."""

    def __init__(
        self, app, limit: Optional[int] = None, window: Optional[int] = None
    ) -> None:
        super().__init__(app)
        self.limit = int(os.environ.get("API_RATE_LIMIT", str(limit or 100)))
        self.window = int(os.environ.get("API_RATE_WINDOW", str(window or 60)))
        self.requests: defaultdict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "anonymous"
        now = time.time()
        window_start = now - self.window
        times = [t for t in self.requests[ip] if t > window_start]
        if len(times) >= self.limit:
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        times.append(now)
        self.requests[ip] = times
        return await call_next(request)
