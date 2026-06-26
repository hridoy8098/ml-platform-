from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from config import settings


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if settings.API_KEY is None:
            return await call_next(request)

        if request.url.path in ("/health", "/", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        api_key = request.headers.get(settings.API_KEY_HEADER)
        if not api_key or api_key != settings.API_KEY:
            return JSONResponse(
                status_code=401,
                content={"success": False, "error": "UNAUTHORIZED",
                         "message": "Valid API key required. Set X-API-Key header."}
            )
        return await call_next(request)
