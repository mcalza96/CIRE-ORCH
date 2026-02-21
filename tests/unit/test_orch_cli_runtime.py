import asyncio

from app import orch_cli_runtime


def test_dispatch_chat_command(monkeypatch):
    calls: dict[str, list[str]] = {}

    async def _chat_main(argv=None):
        calls["argv"] = list(argv or [])

    monkeypatch.setattr(orch_cli_runtime.chat_cli_runtime, "main", _chat_main)

    code = orch_cli_runtime._main(["chat", "--doctor"])
    assert code == 0
    assert calls["argv"] == ["--doctor"]


def test_dispatch_ingest_command(monkeypatch):
    async def _build(args, env=None):
        return orch_cli_runtime.ExecContext(
            script="/tmp/ing.sh",
            argv=["/tmp/ing.sh", "--help"],
            env={"A": "1"},
        )

    called = {}

    def _exec(script, argv, env):
        called["script"] = script
        called["argv"] = argv
        called["env"] = env

    monkeypatch.setattr(orch_cli_runtime, "build_ingest_exec_context", _build)

    code = orch_cli_runtime._main(["ingest", "--help"], exec_fn=_exec)
    assert code == 0
    assert called["script"] == "/tmp/ing.sh"
    assert called["argv"] == ["/tmp/ing.sh", "--help"]
    assert called["env"] == {"A": "1"}


def test_resolve_access_token_precedence(monkeypatch):
    async def _should_not_run(*args, **kwargs):  # pragma: no cover
        raise AssertionError("ensure_access_token should not be called")

    monkeypatch.setattr(orch_cli_runtime, "ensure_access_token", _should_not_run)

    token = asyncio.run(
        orch_cli_runtime._resolve_access_token(
            env={
                "ORCH_ACCESS_TOKEN": "orch-token",
                "SUPABASE_ACCESS_TOKEN": "supa-token",
                "AUTH_BEARER_TOKEN": "bearer-token",
            },
            non_interactive=False,
        )
    )
    assert token == "orch-token"


def test_resolve_access_token_skips_expired_env_token(monkeypatch):
    async def _should_not_run(*args, **kwargs):  # pragma: no cover
        raise AssertionError("ensure_access_token should not be called")

    monkeypatch.setattr(orch_cli_runtime, "ensure_access_token", _should_not_run)
    monkeypatch.setattr(
        orch_cli_runtime,
        "decode_jwt_exp",
        lambda token: 0 if token == "expired-token" else None,
    )

    token = asyncio.run(
        orch_cli_runtime._resolve_access_token(
            env={
                "ORCH_ACCESS_TOKEN": "expired-token",
                "SUPABASE_ACCESS_TOKEN": "fresh-token",
            },
            non_interactive=False,
        )
    )
    assert token == "fresh-token"


def test_resolve_access_token_fallback(monkeypatch):
    seen = {}

    async def _ensure_access_token(*, interactive):
        seen["interactive"] = interactive
        return "resolved-token"

    monkeypatch.setattr(orch_cli_runtime, "ensure_access_token", _ensure_access_token)

    token = asyncio.run(
        orch_cli_runtime._resolve_access_token(
            env={},
            non_interactive=True,
        )
    )
    assert token == "resolved-token"
    assert seen["interactive"] is False


def test_resolve_rag_url_force_and_probe(monkeypatch):
    env = {
        "RAG_ENGINE_LOCAL_URL": "http://local:8000",
        "RAG_ENGINE_DOCKER_URL": "http://docker:8000",
    }

    monkeypatch.setattr(orch_cli_runtime, "_rag_local_healthy", lambda **kwargs: True)
    assert orch_cli_runtime._resolve_rag_url(env) == "http://local:8000"

    monkeypatch.setattr(orch_cli_runtime, "_rag_local_healthy", lambda **kwargs: False)
    assert orch_cli_runtime._resolve_rag_url(env) == "http://docker:8000"

    env_force = dict(env)
    env_force["RAG_ENGINE_FORCE_BACKEND"] = "docker"
    assert orch_cli_runtime._resolve_rag_url(env_force) == "http://docker:8000"


def test_build_ingest_context_non_interactive_flag(monkeypatch):
    base_dir = orch_cli_runtime._repo_root()
    assert (base_dir / "tools/ingestion-client/ing.sh").exists()

    monkeypatch.setattr(orch_cli_runtime, "_require_orch_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(orch_cli_runtime, "_resolve_rag_url", lambda _env: "http://resolved-rag:8000")

    calls = {}

    async def _ensure_access_token(*, interactive):
        calls["interactive"] = interactive
        return "resolved-token"

    monkeypatch.setattr(orch_cli_runtime, "ensure_access_token", _ensure_access_token)
    async def _tenant_none(**_kwargs):
        return None, False

    monkeypatch.setattr(orch_cli_runtime, "_resolve_tenant_from_orch", _tenant_none)

    context = asyncio.run(
        orch_cli_runtime.build_ingest_exec_context(
            ["--orch-non-interactive-auth", "--no-wait"],
            env={"ORCH_URL": "http://localhost:8001"},
        )
    )

    assert calls["interactive"] is False
    assert context.env["ORCH_ACCESS_TOKEN"] == "resolved-token"
    assert context.env["RAG_URL"] == "http://resolved-rag:8000"
    assert context.argv[-1] == "--no-wait"


def test_build_ingest_context_help_skips_bootstrap(monkeypatch):
    called = {"health": False, "token": False}

    def _health(*_args, **_kwargs):
        called["health"] = True

    async def _token(*_args, **_kwargs):
        called["token"] = True
        return "token"

    monkeypatch.setattr(orch_cli_runtime, "_require_orch_health", _health)
    monkeypatch.setattr(orch_cli_runtime, "_resolve_access_token", _token)

    context = asyncio.run(orch_cli_runtime.build_ingest_exec_context(["--help"], env={}))
    assert context.argv[-1] == "--help"
    assert called["health"] is False
    assert called["token"] is False


def test_build_ingest_context_injects_tenant_from_discovery(monkeypatch):
    base_dir = orch_cli_runtime._repo_root()
    assert (base_dir / "tools/ingestion-client/ing.sh").exists()

    monkeypatch.setattr(orch_cli_runtime, "_require_orch_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(orch_cli_runtime, "_resolve_rag_url", lambda _env: "http://resolved-rag:8000")
    async def _resolve_token(**_kwargs):
        return "resolved-token"

    monkeypatch.setattr(orch_cli_runtime, "_resolve_access_token", _resolve_token)

    async def _tenant(**_kwargs):
        return orch_cli_runtime.Tenant(id="tenant-1", name="Tenant Uno"), False

    monkeypatch.setattr(orch_cli_runtime, "_resolve_tenant_from_orch", _tenant)

    context = asyncio.run(
        orch_cli_runtime.build_ingest_exec_context(
            ["--no-wait"],
            env={"ORCH_URL": "http://localhost:8001"},
        )
    )
    assert context.env["TENANT_ID"] == "tenant-1"
    assert context.env["TENANT_NAME"] == "Tenant Uno"


def test_build_ingest_context_respects_explicit_tenant(monkeypatch):
    base_dir = orch_cli_runtime._repo_root()
    assert (base_dir / "tools/ingestion-client/ing.sh").exists()

    monkeypatch.setattr(orch_cli_runtime, "_require_orch_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(orch_cli_runtime, "_resolve_rag_url", lambda _env: "http://resolved-rag:8000")
    async def _resolve_token(**_kwargs):
        return "resolved-token"

    monkeypatch.setattr(orch_cli_runtime, "_resolve_access_token", _resolve_token)

    called = {"discovery": False}

    async def _tenant(**_kwargs):
        called["discovery"] = True
        return None, False

    monkeypatch.setattr(orch_cli_runtime, "_resolve_tenant_from_orch", _tenant)

    context = asyncio.run(
        orch_cli_runtime.build_ingest_exec_context(
            ["--tenant-id", "manual-tenant"],
            env={"ORCH_URL": "http://localhost:8001"},
        )
    )
    assert "TENANT_ID" not in context.env
    assert called["discovery"] is False


def test_build_ingest_context_preserves_agent_profile_flags(monkeypatch):
    base_dir = orch_cli_runtime._repo_root()
    assert (base_dir / "tools/ingestion-client/ing.sh").exists()

    monkeypatch.setattr(orch_cli_runtime, "_require_orch_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(orch_cli_runtime, "_resolve_rag_url", lambda _env: "http://resolved-rag:8000")

    async def _resolve_token(**_kwargs):
        return "resolved-token"

    monkeypatch.setattr(orch_cli_runtime, "_resolve_access_token", _resolve_token)

    context = asyncio.run(
        orch_cli_runtime.build_ingest_exec_context(
            ["--agent-profile", "iso_auditor", "--no-wait"],
            env={"ORCH_URL": "http://localhost:8001", "TENANT_ID": "tenant-1"},
        )
    )
    assert "--agent-profile" in context.argv
    assert "iso_auditor" in context.argv


def test_resolve_tenant_from_orch_fallback_parses_list_payload(monkeypatch):
    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return [{"tenant_id": "t-fallback", "name": "Tenant Fallback"}]

    async def _empty_tenants(*_args, **_kwargs):
        return []

    monkeypatch.setattr(orch_cli_runtime, "list_authorized_tenants", _empty_tenants)
    monkeypatch.setattr(orch_cli_runtime.httpx, "get", lambda *args, **kwargs: _Response())

    tenant, unauthorized = asyncio.run(
        orch_cli_runtime._resolve_tenant_from_orch(
            orchestrator_url="http://localhost:8001",
            token="token",
            non_interactive=True,
        )
    )
    assert tenant is not None
    assert unauthorized is False
    assert tenant.id == "t-fallback"


def test_build_ingest_context_retries_token_after_401(monkeypatch):
    base_dir = orch_cli_runtime._repo_root()
    assert (base_dir / "tools/ingestion-client/ing.sh").exists()

    monkeypatch.setattr(orch_cli_runtime, "_require_orch_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(orch_cli_runtime, "_resolve_rag_url", lambda _env: "http://resolved-rag:8000")

    async def _resolve_token(**_kwargs):
        return "stale-token"

    monkeypatch.setattr(orch_cli_runtime, "_resolve_access_token", _resolve_token)

    state = {"calls": 0}

    async def _tenant(**_kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return None, True
        return orch_cli_runtime.Tenant(id="tenant-2", name="Tenant Dos"), False

    async def _ensure_access_token(*, interactive):
        assert interactive is True
        return "fresh-token"

    monkeypatch.setattr(orch_cli_runtime, "_resolve_tenant_from_orch", _tenant)
    monkeypatch.setattr(orch_cli_runtime, "ensure_access_token", _ensure_access_token)

    context = asyncio.run(
        orch_cli_runtime.build_ingest_exec_context(
            ["--no-wait"],
            env={"ORCH_URL": "http://localhost:8001"},
        )
    )

    assert context.env["ORCH_ACCESS_TOKEN"] == "fresh-token"
    assert context.env["TENANT_ID"] == "tenant-2"
    assert state["calls"] == 2
