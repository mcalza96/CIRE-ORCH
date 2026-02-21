import json
import threading

from app.cartridges.dev_assignments import DevProfileAssignmentsStore, get_dev_profile_assignments_store
from app.infrastructure.config import settings


def _configure_store(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", True)
    monkeypatch.setattr(
        settings,
        "ORCH_DEV_PROFILE_ASSIGNMENTS_FILE",
        str(tmp_path / "tenant_profile_assignments.json"),
    )
    get_dev_profile_assignments_store.cache_clear()
    return get_dev_profile_assignments_store()


def test_store_set_get_persists_to_disk(monkeypatch, tmp_path):
    store = _configure_store(monkeypatch, tmp_path)
    store.set("tenant-a", "iso_auditor")

    assert store.get("tenant-a") == "iso_auditor"
    assert store.path.exists()

    second = DevProfileAssignmentsStore(path=store.path)
    assert second.get("tenant-a") == "iso_auditor"


def test_store_clear_removes_assignment(monkeypatch, tmp_path):
    store = _configure_store(monkeypatch, tmp_path)
    store.set("tenant-a", "iso_auditor")

    assert store.clear("tenant-a") is True
    assert store.get("tenant-a") is None
    assert store.clear("tenant-a") is False


def test_store_snapshot_ignores_invalid_rows(monkeypatch, tmp_path):
    store = _configure_store(monkeypatch, tmp_path)
    payload = {
        "tenant-a": "iso_auditor",
        "tenant-b": "",
        " ": "legal_cl",
        "tenant-c": None,
    }
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps(payload), encoding="utf-8")

    assert store.snapshot() == {"tenant-a": "iso_auditor"}


def test_store_thread_safe_writes(monkeypatch, tmp_path):
    store = _configure_store(monkeypatch, tmp_path)

    def _writer(index: int) -> None:
        store.set(f"tenant-{index}", "base")

    threads = [threading.Thread(target=_writer, args=(idx,)) for idx in range(25)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = store.snapshot()
    assert len(snapshot) == 25
    assert snapshot["tenant-0"] == "base"
