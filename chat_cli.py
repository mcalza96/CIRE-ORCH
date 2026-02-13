"""CLI entrypoint for orchestrator extraction staging."""

from __future__ import annotations

import asyncio

from runtime.chat_cli_runtime import main


if __name__ == "__main__":
    asyncio.run(main())
