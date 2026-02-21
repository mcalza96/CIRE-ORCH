#!/usr/bin/env bash
set -euo pipefail

# Uso:
# TENANT_ID="..." COLLECTION_ID="..." ./diag_orch_ingesta.sh
# Opcional:
# ORCH_URL=http://localhost:8001
# QUERY="que es la iso 9001?"
# PROFILE_ID=base
# ACCESS_TOKEN="..."   # si no se define, intenta leer ~/.cire/orch/session.json

: "${TENANT_ID:?TENANT_ID requerido}"
ORCH_URL="${ORCH_URL:-http://localhost:8001}"
QUERY="${QUERY:-que es la iso 9001?}"
PROFILE_ID="${PROFILE_ID:-base}"
COLLECTION_ID="${COLLECTION_ID:-}"

if [[ -z "${ACCESS_TOKEN:-}" ]]; then
  ACCESS_TOKEN="$(python3 - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.path.expanduser("~/.cire/orch/session.json"))
if not p.exists():
    print("")
    raise SystemExit(0)
try:
    data = json.loads(p.read_text(encoding="utf-8"))
    print((data.get("access_token") or "").strip())
except Exception:
    print("")
PY
)"
fi

if [[ -z "${ACCESS_TOKEN}" ]]; then
  echo "ERROR: No ACCESS_TOKEN. Exporta ACCESS_TOKEN o inicia ./chat.sh una vez."
  exit 1
fi

export ORCH_URL TENANT_ID QUERY PROFILE_ID COLLECTION_ID ACCESS_TOKEN

python3 - <<'PY'
import json, os, urllib.request, urllib.error

orch = os.environ["ORCH_URL"].rstrip("/")
tenant = os.environ["TENANT_ID"]
query = os.environ.get("QUERY", "que es la iso 9001?")
profile = os.environ.get("PROFILE_ID", "base")
collection = os.environ.get("COLLECTION_ID", "").strip()
token = os.environ["ACCESS_TOKEN"]


def post(path, payload, profile_id=None):
    url = f"{orch}{path}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-Tenant-ID": tenant,
    }
    if profile_id:
        headers["X-Agent-Profile"] = profile_id
    req = urllib.request.Request(
        url=url,
        method="POST",
        headers=headers,
        data=json.dumps(payload).encode("utf-8"),
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"\nHTTP {e.code} {path}\n{body}\n")
        raise


def explain(filters):
    payload = {
        "query": query,
        "tenant_id": tenant,
        "collection_id": collection or None,
        "top_n": 10,
        "k": 12,
        "fetch_k": 60,
        "filters": filters,
    }
    data = post("/api/v1/knowledge/explain-retrieval", payload, profile_id=profile)
    items = data.get("items") if isinstance(data.get("items"), list) else []
    penalized = 0
    for it in items:
        comps = (((it.get("explain") or {}).get("score_components") or {}))
        if bool(comps.get("scope_penalized")):
            penalized += 1
    print(f"- explain filters={filters}: items={len(items)} scope_penalized={penalized}/{len(items)}")
    if items:
        top = items[0]
        comps = (((top.get("explain") or {}).get("score_components") or {}))
        print(f"  top1 source={top.get('source')} score={top.get('score')} final={comps.get('final_score')}")


def answer(profile_id):
    payload = {
        "query": query,
        "tenant_id": tenant,
        "collection_id": collection or None,
    }
    data = post("/api/v1/knowledge/answer", payload, profile_id=profile_id)
    ctx = data.get("context_chunks") if isinstance(data.get("context_chunks"), list) else []
    cites = data.get("citations") if isinstance(data.get("citations"), list) else []
    val = data.get("validation") if isinstance(data.get("validation"), dict) else {}
    ret = data.get("retrieval") if isinstance(data.get("retrieval"), dict) else {}
    trace = ret.get("trace") if isinstance(ret.get("trace"), dict) else {}
    print(
        f"- answer profile={profile_id}: mode={data.get('mode')} ctx={len(ctx)} cites={len(cites)} accepted={val.get('accepted')}"
    )
    if val.get("issues"):
        print(f"  issues={val.get('issues')}")
    if data.get("requested_scopes"):
        print(f"  requested_scopes={data.get('requested_scopes')}")
    miss = trace.get("missing_scopes")
    if miss:
        print(f"  missing_scopes={miss}")
    warns = trace.get("warnings")
    if warns:
        print(f"  warnings={warns[:3]}")
    ans = (data.get("answer") or "").strip().replace("\n", " ")
    print(f"  answer_preview={ans[:180]}")


print("\n== EXPLAIN DIAGNOSTIC ==")
explain({"source_standard": "ISO 9001"})
explain({"source_standard": "9001"})
explain(None)

print("\n== ANSWER DIAGNOSTIC ==")
answer(os.environ["PROFILE_ID"])
if os.environ["PROFILE_ID"] != "iso_auditor":
    answer("iso_auditor")

print("\nOK")
PY
