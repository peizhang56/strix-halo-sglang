#!/usr/bin/env bash
# Launch a Claude Code session against the LOCAL sglang server instead of the corporate
# gateway (https://llm-api.amd.com/Anthropic).
#
# There is no way to retarget a Claude Code session that is already running, and no way
# for one session to retarget another: the endpoint is read from the environment at
# process start. So this script sets the environment and execs a NEW claude.
#
# Run it from a normal shell, not from inside an existing Claude Code session.
#
#   ./claude_local.sh                       # interactive session on the local model
#   ./claude_local.sh -p 'say hi'           # one-shot, good for a first smoke test
#   CLAUDE_LOCAL_MODEL=... ./claude_local.sh
#   SGLANG_HOST=localhost ./claude_local.sh # from the host, port 30000 is published
set -euo pipefail

HOSTNAME_="${SGLANG_HOST:-127.0.0.1}"
PORT="${SGLANG_PORT:-30000}"
BASE="http://${HOSTNAME_}:${PORT}"

# SGLang ignores the request's `model` field and serves whatever was loaded, so this name
# is only for readability in logs. Do NOT append the `[1m]` suffix seen in upstream
# examples -- it turns on Claude Code's 1M-context beta and will overrun --context-length.
MODEL="${CLAUDE_LOCAL_MODEL:-qwen3-27b-int4}"

# Claude Code does not know this model name, so it assumes a 200k window and would let the
# conversation grow past whatever --context-length the server was started with. Tell it
# the truth. Keep in sync with SGLANG_CONTEXT_LEN in sglang_server.sh.
CONTEXT_LEN="${SGLANG_CONTEXT_LEN:-65536}"

# Claude Code puts an effort level in output_config on every request; SGLang forwards it
# to the chat template unchanged. Qwen3.5's template accepts only xhigh/medium/low, so the
# default "high" raises jinja2 TemplateError and the server answers 500 BEFORE any
# inference runs. Claude Code reports that as "500 Internal server error ... usually
# temporary", which is misleading -- it is deterministic and it is a config mismatch.
# This has to be the --effort flag; there is no env var for it.
EFFORT="${CLAUDE_LOCAL_EFFORT:-low}"

# --- readiness gate -----------------------------------------------------------------
# Two probes, because no single one separates the three states that matter.
#
# /model_info is answered entirely in the HTTP layer and never reaches the scheduler --
# measured 0.011 s while the scheduler was saturated mid-prefill. So it is the honest
# "is a server process listening" test. It is NOT a readiness test: it returns 200
# during warmup, while the model still cannot serve.
#
# /health is the readiness test (503 for the whole warmup), but it is NOT cheap -- it
# pushes a real generate request through the scheduler and waits on the detokenizer. On
# this box one 8192-token prefill chunk takes ~150 s, so a single 22k-token turn from
# another session blocks /health for MINUTES. The server's own log says so:
#   "Health check failed. Server couldn't get a response from detokenizer for last 20s"
# while it is happily prefilling. That is busy, not broken.
#
# Crucially, curl reports %{http_code} as 000 for BOTH a refused connection and a
# timeout, so the status code alone cannot tell "nothing is running" from "running and
# busy" -- the earlier version of this gate conflated them and told you to launch a
# second server on an occupied port. The exit code does distinguish them: 7 vs 28.
HEALTH_TIMEOUT="${CLAUDE_LOCAL_HEALTH_TIMEOUT:-20}"

if ! curl -s -o /dev/null --max-time 5 "${BASE}/model_info" 2>/dev/null; then
    echo "No server is listening at ${BASE}." >&2
    echo "Start it with ./sglang_server.sh and wait for /health to return 200." >&2
    exit 1
fi

# `code=$(curl ...)` on its own would trip errexit the moment curl times out -- the
# assignment takes curl's exit status -- and the script would die before reading $?.
# The `&& / ||` tail suspends errexit for the whole list and captures the code.
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time "$HEALTH_TIMEOUT" "${BASE}/health" 2>/dev/null) && rc=0 || rc=$?
case "$code:$rc" in
    200:*)
        ;;
    503:*)
        echo "Server at ${BASE} is still warming up (/health -> 503)." >&2
        echo "Wait for 200 before launching; traffic during warmup can trip the warmup timeout." >&2
        exit 1
        ;;
    000:28)
        # Up, listening, but the scheduler did not get to our probe within the timeout.
        echo "note: server is up but busy (/health did not answer in ${HEALTH_TIMEOUT}s)." >&2
        echo "      Another request is mid-prefill; your first turn will queue behind it." >&2
        ;;
    *)
        echo "Server at ${BASE} answered /health unexpectedly (http=${code:-none} curl=${rc})." >&2
        exit 1
        ;;
esac

# --- override the endpoint ----------------------------------------------------------
# Exporting ANTHROPIC_BASE_URL is NOT enough. ~/.claude.json carries an `env` block
# pinning it to the corporate gateway, and a settings-file `env` block WINS over the
# process environment -- so plain `export` is silently ignored and every request still
# goes to the gateway. Verified: pointed at a closed port via export, Claude Code still
# answered normally. The override has to come in at a higher precedence level, and
# `--settings` (command line) outranks the user config. Writing the env block into a
# settings file is therefore the mechanism, not a convenience.
SETTINGS="${TMPDIR:-/tmp}/claude_local_settings.json"
cat > "$SETTINGS" <<EOF
{
  "env": {
    "ANTHROPIC_BASE_URL": "${BASE}",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_CUSTOM_HEADERS": "",
    "ANTHROPIC_MODEL": "${MODEL}",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "${MODEL}",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "${MODEL}",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "${MODEL}",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "${CONTEXT_LEN}",
    "API_TIMEOUT_MS": "${API_TIMEOUT_MS:-3000000}",
    "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": "${CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS:-1800000}",
    "CLAUDE_CODE_CONNECT_TIMEOUT_MS": "${CLAUDE_CODE_CONNECT_TIMEOUT_MS:-60000}",
    "CLAUDE_CODE_MAX_RETRIES": "${CLAUDE_CODE_MAX_RETRIES:-1}"
  }
}
EOF

# All four model slots, or background work (titles, summaries) still asks for a Claude-*
# name. SGLang ignores the `model` field and serves whatever was loaded, so this only
# affects readability -- but a stray Claude-* name in the logs is how you fool yourself
# into thinking the gateway is still in play.
#
# ANTHROPIC_CUSTOM_HEADERS is blanked because it carries the APIM subscription key; there
# is no reason to ship that to a local process.
#
# CLAUDE_CODE_ATTRIBUTION_HEADER=0: Claude Code otherwise stamps a per-request hash into
# the system prompt, changing the prefix every turn and defeating radix prefix caching --
# a full re-prefill of the whole history each turn, which this box cannot afford.
# CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC does NOT cover it; separate code path.
#
# API_TIMEOUT_MS: the default fires long before a 27B model on an iGPU finishes a turn.
#
# CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS is a separate budget from API_TIMEOUT_MS -- they
# measure different things, and API_TIMEOUT_MS does NOT cover this one. SGLang emits no SSE
# bytes at all until prefill finishes, so the socket is open and completely silent for the
# whole prefill. When that silence runs out the client aborts with
# "stream idle: no bytes for <N>ms" and retries; the interactive UI shows it as
# "check your network", which is why it looks like a network fault when nothing is wrong.
#
# Measured, not assumed -- a stub endpoint that accepts the request, sends headers, then
# deliberately withholds the first byte:
#   default (variable unset)          client aborted at exactly 300.0 s
#   CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS=20000   client aborted at exactly 20.0 s
# So the default is 300 s, and an ordinary 6,700-token turn (~110 s of prefill) is well
# inside it. The retries seen here were NOT the default being too tight: a second session
# was prefilling concurrently, and with --max-running-requests 2 the queued turn stayed
# silent past 300 s. Raising this is cheap insurance against that, not the primary fix --
# the primary fix is trap 5 (the prefix cache) and not running two sessions at once.
#
# Note the retry happens even at CLAUDE_CODE_MAX_RETRIES=0: the stub logged three attempts.
# This path does not appear to honour that variable.
#
# The retry is worse than the wait, because it is self-reinforcing:
#   - the original request is still running; the server does not know you gave up
#   - the retry arrives before the original has committed its KV to the radix tree, so
#     it ALSO misses the cache and re-prefills all 6,700 tokens from zero
#   - now two full prefills compete for --max-running-requests 2, so both get slower
#   - longer silence -> another timeout -> another retry
# Observed exactly this: turn 2 logged #cached-token: 0 where it should have hit, while a
# later settled turn on the same prefix logged #new-token: 734, #cached-token: 6528.
# Hence MAX_RETRIES=1 as well -- on this box a retry cannot help, it can only pile on.

# Nested launch: a `claude` started from inside another Claude Code session inherits the
# parent's messaging socket and delegates to it instead of calling the API itself -- so it
# answers correctly while the local server sees zero traffic. Scrub those so this is a
# real, independent session.
unset CLAUDECODE CLAUDE_CODE_MESSAGING_SOCKET CLAUDE_CODE_MESSAGING_TOKEN \
      CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_SESSION_ID CLAUDE_CODE_ENTRYPOINT \
      CLAUDE_CODE_SESSION_ATTENDED CLAUDE_PID

# --- trim the tool schemas ----------------------------------------------------------
# Prefill dominates every turn here, and tool schemas dominate prefill. Measured by
# capturing a real request for the one-word prompt "hi" (21,094 tokens total):
#
#   tool schemas (21 tools)   13,253   63%
#   CLAUDE.md + git reminders  4,436   21%
#   environment + system       3,148   15%
#   the word "hi"                 53
#
# --disallowed-tools removes the schemas from the body, not just the permission list --
# verified by capture: 21 tools 13,253 tokens -> 6 tools 2,395 tokens. That is 10,858
# tokens, ~3.3 minutes of prefill at the 54 tok/s measured here, off every cold turn.
#
# The dropped set is agent/cron/worktree machinery. Agent, Workflow, ScheduleWakeup and
# SendMessage do not just go unused on this box -- they SPAWN MORE MODEL CALLS, and a
# subagent here is another multi-minute prefill. Dropping them is a speedup twice over.
#
#   CLAUDE_LOCAL_TOOLS=all             send the full set (slower, full capability)
#   CLAUDE_LOCAL_DROP_TOOLS='A,B,C'    override the list
LEAN_DROP='Workflow,ScheduleWakeup,SendMessage,CronCreate,CronDelete,CronList,EnterWorktree,ExitWorktree,Agent,ReportFindings,TaskOutput,TaskStop,ListAgents,NotebookEdit,Skill'
DROP_TOOLS="${CLAUDE_LOCAL_DROP_TOOLS:-$LEAN_DROP}"

tool_flags=()
if [ "${CLAUDE_LOCAL_TOOLS:-lean}" = "all" ]; then
    echo "Sending all tool schemas (+~10.9k tokens of prefill per cold turn)."
else
    tool_flags=(--disallowed-tools "$DROP_TOOLS")
fi

echo "Claude Code -> ${BASE} (model: ${MODEL}, effort: ${EFFORT}, tools: ${CLAUDE_LOCAL_TOOLS:-lean})"
# --effort and the tool list go before "$@" so an explicit one from the caller still wins.
exec claude --settings "$SETTINGS" --effort "$EFFORT" \
     ${tool_flags[@]+"${tool_flags[@]}"} "$@"
