from __future__ import annotations

import asyncio

from .service import start_embeddings_task


async def _main() -> None:
    task = start_embeddings_task()
    try:
        await task
    except (KeyboardInterrupt, asyncio.CancelledError):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(_main())
