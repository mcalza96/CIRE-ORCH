import os
import subprocess
from pathlib import Path


def _make_fake_curl(tmp_path: Path) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    curl_path = bin_dir / "curl"
    curl_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(curl_path, 0o755)
    return str(bin_dir)


def test_ing_sh_uses_shared_auth_bridge(tmp_path):
    repo = Path("/Users/mcalzadilla/cire/orch")
    script = repo / "ing.sh"
    content = script.read_text(encoding="utf-8")
    assert "orch_cli.py" in content
    assert "auth_cli_runtime.py" not in content

    fake_bin = _make_fake_curl(tmp_path)
    env = dict(os.environ)
    env["PATH"] = fake_bin + os.pathsep + env.get("PATH", "")
    env["ORCH_URL"] = "http://127.0.0.1:8001"

    proc = subprocess.run(
        [str(script), "--help"],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Uso:" in proc.stdout
    assert "--agent-profile <id>" in proc.stdout
    assert "--clear-agent-profile" in proc.stdout


def test_ing_sh_accepts_agent_profile_flags_in_help_mode(tmp_path):
    repo = Path("/Users/mcalzadilla/cire/orch")
    script = repo / "ing.sh"

    fake_bin = _make_fake_curl(tmp_path)
    env = dict(os.environ)
    env["PATH"] = fake_bin + os.pathsep + env.get("PATH", "")
    env["ORCH_URL"] = "http://127.0.0.1:8001"

    proc = subprocess.run(
        [str(script), "--agent-profile", "iso_auditor", "--help"],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Uso:" in proc.stdout
