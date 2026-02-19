from __future__ import annotations

import argparse
import os
from typing import Final

DOCTOR_DEFAULT_QUERY: Final = "Que exige la referencia 7.5.3 en este alcance?"

def parse_chat_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_orchestrator_url = (
        os.getenv("ORCH_URL") or os.getenv("ORCHESTRATOR_URL") or "http://localhost:8001"
    )
    parser = argparse.ArgumentParser(description="Q/A chat via Orchestrator API")
    parser.add_argument(
        "--tenant-id", help="Institutional tenant id (optional if tenant storage is configured)"
    )
    parser.add_argument(
        "--tenant-storage-path",
        help="Optional path to persisted tenant context JSON",
    )
    parser.add_argument("--collection-id", help="Collection id (optional)")
    parser.add_argument("--collection-name", help="Collection name (display only)")
    parser.add_argument(
        "--agent-profile",
        help="Optional cartridge/profile id for this chat session",
    )
    parser.add_argument(
        "--orchestrator-url",
        default=default_orchestrator_url,
        help="Base URL for orchestrator API",
    )
    parser.add_argument(
        "--access-token",
        default=(
            os.getenv("ORCH_ACCESS_TOKEN")
            or os.getenv("SUPABASE_ACCESS_TOKEN")
            or os.getenv("AUTH_BEARER_TOKEN")
            or ""
        ),
        help="Bearer token for orchestrator auth",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail if auth or tenant selection requires user input",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run auth/discovery/retrieval diagnosis and exit",
    )
    parser.add_argument(
        "--doctor-query",
        default=DOCTOR_DEFAULT_QUERY,
        help="Controlled query used by --doctor",
    )
    parser.add_argument(
        "--obs",
        action="store_true",
        default=os.getenv("ORCH_CHAT_OBS", "1") != "0",
        help="Show compact observability diagnostics after each answer",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("ORCH_HTTP_READ_TIMEOUT_SECONDS") or 45.0),
        help="Read timeout for /knowledge/answer requests (default: env ORCH_HTTP_READ_TIMEOUT_SECONDS or 45)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print verbose error diagnostics (exception types, traces, HTTP payload snippets)",
    )
    return parser.parse_args(argv)
