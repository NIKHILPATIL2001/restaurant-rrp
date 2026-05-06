#!/usr/bin/env bash
# Post-boot smoke test. Asserts API is up and returns valid data.
set -euo pipefail

BASE="${RRP_API_URL:-http://localhost:8000}"
TOMORROW=$(date -v+1d +%Y-%m-%d 2>/dev/null || date -d "tomorrow" +%Y-%m-%d)

echo "=== Smoke test against $BASE ==="

echo "[1] GET /healthz"
STATUS=$(curl -sf "${BASE}/healthz" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
if [ "$STATUS" != "ok" ]; then
  echo "FAIL: /healthz returned status=$STATUS"
  exit 1
fi
echo "PASS: status=ok"

echo "[2] GET /readyz"
curl -sf "${BASE}/readyz" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['db'] == 'ok', f\"db={d['db']}\""
echo "PASS: db=ok"

echo "[3] POST /v1/forecast/covers"
COVERS=$(curl -sf -X POST "${BASE}/v1/forecast/covers" \
  -H "Content-Type: application/json" \
  -d "{\"date\": \"${TOMORROW}\", \"restaurant_id\": 1}")

N=$(echo "$COVERS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['hourly_covers']))")
if [ "$N" != "24" ]; then
  echo "FAIL: expected 24 hourly_covers, got $N"
  exit 1
fi
echo "PASS: 24-element hourly_covers array"

echo "[4] GET /v1/metrics/convergence"
curl -sf "${BASE}/v1/metrics/convergence?surface=covers" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"PASS: convergence series has {d['summary']['n_days']} days\")"

echo ""
echo "=== All smoke tests passed ==="
