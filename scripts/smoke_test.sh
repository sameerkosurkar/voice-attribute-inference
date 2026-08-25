#!/usr/bin/env bash
# End-to-end smoke test against a running service.
#
#     docker compose up -d && ./scripts/smoke_test.sh
#     BASE_URL=http://localhost:8000 ./scripts/smoke_test.sh
#
# Generates its own fixtures, so it needs no network and no dataset download.
# Exercises the three outcomes that matter: a good-quality prediction, graceful
# degradation on noise, and a typed error on malformed input.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
SAMPLES="${SAMPLES:-samples}"
PY="${PY:-python3}"
failures=0

blue()  { printf '\033[0;34m%s\033[0m\n' "$*"; }
green() { printf '\033[0;32m  %s\033[0m\n' "$*"; }
red()   { printf '\033[0;31m  %s\033[0m\n' "$*"; failures=$((failures + 1)); }

need() { command -v "$1" >/dev/null || { echo "missing dependency: $1"; exit 1; }; }
need curl

# Locate a running service container.
#
# `docker compose ps` only finds containers belonging to THIS directory's
# compose project, so it misses a service someone started from another path or
# via plain `docker run`. Falling back to an image-name lookup makes the script
# work in both cases -- which matters, because the fallback exists precisely for
# reviewers who have nothing installed locally.
find_container() {
  command -v docker >/dev/null 2>&1 || return 0
  local cid
  cid="$(docker compose ps -q voice-attributes 2>/dev/null | head -1 || true)"
  if [ -z "$cid" ]; then
    cid="$(docker ps -q --filter ancestor=voice-attributes:latest 2>/dev/null | head -1 || true)"
  fi
  printf '%s' "$cid"
}

# Fixture generation, in preference order. The point is that a reviewer who has
# only run `docker compose up` -- with no venv and possibly no numpy on the host
# -- can still run this. The container already has numpy, espeak-ng and the
# generator, so we borrow them rather than requiring a host toolchain.
generate_fixtures() {
  if "$PY" -c "import numpy" >/dev/null 2>&1; then
    blue "Generating fixtures (host python) ..."
    "$PY" scripts/make_sample_audio.py --outdir "$SAMPLES" >/dev/null
    return 0
  fi

  # NB: no `... | grep -q` here. Under `set -o pipefail`, grep -q closes the pipe
  # as soon as it matches, the upstream command takes SIGPIPE, and the pipeline
  # reports failure even though it succeeded. Capture and test instead.
  local cid=""
  cid="$(find_container)"

  if [ -n "$cid" ]; then
    blue "Host python has no numpy; generating fixtures inside the container ..."
    mkdir -p "$SAMPLES"
    # Stream a tar rather than `docker cp`: the container's /tmp is a tmpfs
    # mount (see docker-compose.yml) and docker cp cannot read from those.
    if docker exec "$cid" python scripts/make_sample_audio.py --outdir /tmp/samples >/dev/null 2>&1 \
       && docker exec "$cid" sh -c 'cd /tmp/samples && tar cf - .' 2>/dev/null \
          | tar xf - -C "$SAMPLES" 2>/dev/null; then
      # Leave nothing behind in the container. These are synthetic fixtures, not
      # caller audio, so this is tidiness rather than a privacy requirement --
      # but PRIVACY.md claims the service writes no audio to disk, and stray
      # .wav files under /tmp would make that claim harder to check.
      docker exec "$cid" rm -rf /tmp/samples >/dev/null 2>&1 || true
      return 0
    fi
  fi

  red "Could not generate fixtures."
  echo "    Either install numpy on the host (make install), or start the"
  echo "    service first (docker compose up -d) so fixtures can be generated"
  echo "    inside the container."
  exit 1
}

# Check for the SPECIFIC fixtures this script uses, not merely "is the directory
# non-empty". The repo commits three representative samples, so `samples/` is
# non-empty on a fresh clone -- an emptiness check would conclude everything was
# present and then fail on the first fixture that is generated rather than
# committed. (It did exactly that; found by simulating a clean clone.)
REQUIRED_FIXTURES="adult_male_clean.wav adult_male_truck_5db.wav \
adult_male_truck_minus10db.wav silence.wav"

fixtures_missing() {
  local f
  for f in $REQUIRED_FIXTURES; do
    [ -f "$SAMPLES/$f" ] || return 0
  done
  return 1
}

if fixtures_missing; then
  generate_fixtures
fi

for f in $REQUIRED_FIXTURES; do
  [ -f "$SAMPLES/$f" ] || { red "fixture $SAMPLES/$f is still missing after generation"; exit 1; }
done

blue "Waiting for $BASE_URL to become ready ..."
for _ in $(seq 1 60); do
  if curl -fsS "$BASE_URL/ready" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS "$BASE_URL/ready" >/dev/null 2>&1 \
  || { red "service never became ready"; exit 1; }
green "ready"

# field <json> <jq-ish path>  -- avoids a hard jq dependency
field() { "$PY" -c "import json,sys;d=json.load(sys.stdin);print(eval('d'+sys.argv[1]))" "$1"; }

check() {  # check <label> <file> <expected-quality> <expect-known|expect-unknown>
  local label="$1" file="$2" want_quality="$3" want_known="$4"
  blue "POST /analyze  <- $label"
  local body
  body="$(curl -fsS -X POST "$BASE_URL/analyze?debug=true" \
            -F "audio=@${file};type=audio/wav")" || { red "request failed"; return; }

  local quality gender age gconf ms
  quality="$(echo "$body" | field "['audio_quality']")"
  gender="$(echo "$body"  | field "['gender']['prediction']")"
  gconf="$(echo "$body"   | field "['gender']['confidence']")"
  age="$(echo "$body"     | field "['age_bracket']['prediction']")"
  ms="$(echo "$body"      | field "['processing_ms']")"

  echo "    quality=$quality gender=$gender($gconf) age=$age ${ms}ms"

  [ "$quality" = "$want_quality" ] \
    && green "quality is '$want_quality' as expected" \
    || red "expected quality '$want_quality', got '$quality'"

  if [ "$want_known" = "expect-known" ]; then
    [ "$gender" != "unknown" ] \
      && green "returned a gender prediction" \
      || red "expected a gender prediction, got unknown"
  else
    [ "$gender" = "unknown" ] \
      && green "correctly abstained" \
      || red "expected unknown on unusable audio, got '$gender'"
  fi

  [ "$ms" -lt 500 ] \
    && green "processing_ms ${ms} is within the 500 ms target" \
    || red "processing_ms ${ms} exceeds the 500 ms target"
}

check "clean speech"            "$SAMPLES/adult_male_clean.wav"          good         expect-known
check "truck cab, 5 dB SNR"     "$SAMPLES/adult_male_truck_5db.wav"      degraded     expect-known
check "truck cab, -10 dB SNR"   "$SAMPLES/adult_male_truck_minus10db.wav" insufficient expect-unknown
check "dead air"                "$SAMPLES/silence.wav"                   insufficient expect-unknown

blue "POST /analyze  <- garbage bytes (expect 415)"
code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/analyze" \
          -H 'content-type: audio/wav' --data-binary 'definitely not audio')"
[ "$code" = "415" ] && green "got 415 as expected" || red "expected 415, got $code"

# The streaming endpoint. Kept in the smoke test because the REST checks above
# passed happily through a refactor that never touched -- and never exercised --
# the WebSocket route. `websockets` is not stdlib, so run it wherever it exists:
# the host if possible, otherwise the container, which has it.
blue "WS /ws/analyze"
ws_host="$(echo "$BASE_URL" | sed -E 's#^https?://##')"
if "$PY" -c "import websockets" >/dev/null 2>&1; then
  if "$PY" scripts/ws_smoke.py --url "ws://${ws_host}/ws/analyze" --samples "$SAMPLES"; then
    green "streaming endpoint ok"
  else
    red "streaming endpoint failed"
  fi
elif cid="$(find_container)"; [ -n "$cid" ]; then
  docker exec "$cid" mkdir -p /tmp/samples >/dev/null 2>&1 || true
  tar cf - -C "$SAMPLES" . 2>/dev/null | docker exec -i "$cid" tar xf - -C /tmp/samples 2>/dev/null || true
  if docker exec "$cid" python scripts/ws_smoke.py \
        --url "ws://127.0.0.1:8000/ws/analyze" --samples /tmp/samples; then
    green "streaming endpoint ok (checked inside the container)"
  else
    red "streaming endpoint failed"
  fi
  docker exec "$cid" rm -rf /tmp/samples >/dev/null 2>&1 || true
else
  echo "    (skipped: no 'websockets' on the host and no running container)"
fi

blue "GET /metrics"
curl -fsS "$BASE_URL/metrics" | grep -q va_request_duration_seconds \
  && green "prometheus metrics exposed" || red "metrics missing"

echo
if [ "$failures" -eq 0 ]; then
  printf '\033[0;32mSMOKE TEST PASSED\033[0m\n'
else
  printf '\033[0;31mSMOKE TEST FAILED (%d checks)\033[0m\n' "$failures"
  exit 1
fi
