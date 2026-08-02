import asyncio

from app.core.config import settings


agent_request_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
