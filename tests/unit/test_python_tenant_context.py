import json

import pytest

from sdk.python.cire_rag_sdk.client import TenantContext


def test_tenant_context_initializes_empty_without_storage():
    context = TenantContext()
    assert context.get_tenant() is None


def test_tenant_context_set_and_get():
    context = TenantContext()
    context.set_tenant("tenant-123")
    assert context.get_tenant() == "tenant-123"


def test_tenant_context_rejects_invalid_tenant():
    context = TenantContext()
    with pytest.raises(ValueError):
        context.set_tenant("  ")


def test_tenant_context_persists_and_restores(tmp_path):
    storage = tmp_path / "tenant-context.json"
    writer = TenantContext(storage_path=storage)
    writer.set_tenant("tenant-persisted")

    reader = TenantContext(storage_path=storage)
    assert reader.get_tenant() == "tenant-persisted"


def test_tenant_context_ignores_corrupted_storage(tmp_path):
    storage = tmp_path / "tenant-context.json"
    storage.write_text("{invalid json", encoding="utf-8")
    context = TenantContext(storage_path=storage)
    assert context.get_tenant() is None


def test_tenant_context_reload_reads_updated_file(tmp_path):
    storage = tmp_path / "tenant-context.json"
    context = TenantContext(tenant_id="tenant-a", storage_path=storage)
    storage.write_text(json.dumps({"tenant_id": "tenant-b"}), encoding="utf-8")

    assert context.reload() == "tenant-b"
    assert context.get_tenant() == "tenant-b"
