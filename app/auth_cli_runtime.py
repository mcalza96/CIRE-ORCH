from __future__ import annotations

import argparse
import asyncio

from app.infrastructure.clients.auth_client import (
    AuthClientError,
    ensure_access_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auth helper runtime for ORCH scripts")
    parser.add_argument(
        "--print-access-token",
        action="store_true",
        help="Print a valid access token to stdout",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail instead of prompting for credentials",
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    if not args.print_access_token:
        return 0
    try:
        token = await ensure_access_token(interactive=not args.non_interactive)
    except AuthClientError as exc:
        print(f"❌ {exc}")
        return 1
    print(token)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
