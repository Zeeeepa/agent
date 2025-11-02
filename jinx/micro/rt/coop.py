from __future__ import annotations

import asyncio
import os


async def coop() -> None:
    """Cooperative yield to keep UI responsive.

    Controlled by env JINX_COOP_YIELD (default: on).
    """
    try:
        on = str(os.getenv("JINX_COOP_YIELD", "1")).strip().lower() not in ("", "0", "false", "off", "no")
    except Exception:
        on = True
    if not on:
        return
    try:
        await asyncio.sleep(0)
    except Exception:
        pass
