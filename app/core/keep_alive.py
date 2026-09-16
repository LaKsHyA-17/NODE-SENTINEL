import asyncio
import logging
import os
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


async def start_keep_alive_loop():
    """
    Self-ping keep-alive background worker for cloud hosting platforms like Render.
    Prevents free-tier instances from spinning down / rebooting due to 15-minute inactivity timeouts.
    """
    url = settings.KEEP_ALIVE_URL or os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("APP_URL")
    if not url:
        logger.info("Keep-alive worker idle: No KEEP_ALIVE_URL, RENDER_EXTERNAL_URL, or APP_URL configured.")
        return

    # Normalize url (remove trailing slash)
    target_url = url.rstrip("/")
    if not target_url.startswith("http://") and not target_url.startswith("https://"):
        target_url = f"https://{target_url}"

    health_endpoint = f"{target_url}/health"
    interval = max(60, getattr(settings, "KEEP_ALIVE_INTERVAL_SECONDS", 600))

    logger.info(f"Keep-alive worker initialized for {health_endpoint} (interval: {interval}s / {interval//60}m).")

    # Initial delay to ensure the server is fully started and listening
    await asyncio.sleep(45)

    async with httpx.AsyncClient(timeout=15.0) as client:
        while True:
            try:
                response = await client.get(health_endpoint)
                if response.status_code == 200:
                    logger.info(f"Keep-alive ping successful: {health_endpoint} -> 200 OK")
                else:
                    logger.warning(f"Keep-alive ping returned status {response.status_code} for {health_endpoint}")
            except asyncio.CancelledError:
                logger.info("Keep-alive background task cancelled.")
                break
            except Exception as exc:
                logger.warning(f"Keep-alive ping failed for {health_endpoint}: {exc}")

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logger.info("Keep-alive background task cancelled during sleep.")
                break
