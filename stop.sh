#!/usr/bin/env bash
# Stop the SGLang server started by ./sglang_server.sh.
#
# Shuts the tree down parent-first so `launch_server` gets the chance to terminate and
# reap its own sglang::scheduler / sglang::detokenizer children. Killing the whole set at
# once (the previous behaviour) tends to drop the parent before its children, which
# reparents them to PID 1 -- and PID 1 in this container is `sleep infinity`, which never
# calls wait(), so they linger as zombies. Graceful SIGTERM first so the scheduler can
# release the weights and KV pool cleanly; SIGKILL only if it refuses to drain.
#
# Safe to run when nothing is up (exits 0).
#
#   ./stop.sh              stop, verify, report
#   SGLANG_PORT=30001      override the port to check
#   SGLANG_STOP_GRACE=30   seconds to wait before escalating to SIGKILL
#
# Deliberately not `set -e`: pgrep exits 1 when there is no match, which is a normal
# outcome here, not an error.
set -uo pipefail

PORT="${SGLANG_PORT:-30000}"
GRACE="${SGLANG_STOP_GRACE:-20}"

# The launcher, and separately the children it forks. Matching the children explicitly
# means a half-dead tree still gets cleaned up.
collect_parents()  { pgrep -f 'sglang\.launch_server' 2>/dev/null | sort -un; }
collect_children() { pgrep -f '^sglang::'             2>/dev/null | sort -un; }
collect_pids()     { { collect_parents; collect_children; } | sort -un; }

# Those two patterns do NOT cover the whole tree. A live server here also has:
#   - multiprocessing.resource_tracker   (child of the launcher)
#   - torch/_inductor/compile_worker/*   (children of sglang::scheduler)
# all of which report comm=python3 with no "sglang" anywhere in argv. They are the ones
# that end up orphaned -- when the scheduler dies its compile workers are reparented to
# PID 1 -- so walk the real process tree instead of trusting the name patterns.
# Snapshot it BEFORE signalling anything: once a parent dies its children are reparented
# and are no longer discoverable as descendants.
collect_tree() {
    local roots=() frontier=() next=() all=() pid ppid c
    mapfile -t roots < <(collect_pids)
    [ ${#roots[@]} -eq 0 ] && return 0

    declare -A kids=()
    while read -r pid ppid; do
        kids[$ppid]="${kids[$ppid]:-} $pid"
    done < <(ps -eo pid=,ppid= 2>/dev/null)

    frontier=("${roots[@]}")
    while [ ${#frontier[@]} -gt 0 ]; do
        next=()
        for pid in "${frontier[@]}"; do
            all+=("$pid")
            for c in ${kids[$pid]:-}; do next+=("$c"); done
        done
        frontier=(${next[@]+"${next[@]}"})
    done
    # Dedup, preserving breadth-first order so callers can signal deepest-last.
    printf '%s\n' "${all[@]}" | awk '!seen[$0]++'
}

# "Live" means exists AND is not a zombie. A zombie has already exited: it holds no
# memory, no GPU handle and no port, and cannot be signalled. Counting one as still
# running would spin every drain loop below until it timed out and then escalate to a
# SIGKILL that can never land.
is_live() {
    local st
    st=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
    [ -n "$st" ] || return 1
    case "$st" in Z*) return 1 ;; esac
    return 0
}

# Guard against PID reuse between the snapshot and the kill: only signal a PID that still
# looks like part of this server. Cheap insurance -- the escalation path sends SIGKILL.
still_ours() {
    local cl
    cl=$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null) || return 1
    case "$cl" in
        *sglang.launch_server*|*sglang::*|*resource_tracker*|*compile_worker*) return 0 ;;
    esac
    grep -q '^sglang' "/proc/$1/comm" 2>/dev/null
}

# Of the snapshotted tree, which members are still live and still ours.
survivors() {
    local pid
    for pid in ${TREE[@]+"${TREE[@]}"}; do
        is_live "$pid" && still_ours "$pid" && printf '%s\n' "$pid"
    done
}

# No ss/netstat/lsof in this image, so probe the socket directly.
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; }

count_zombies() { ps -eo stat= 2>/dev/null | grep -c '^Z'; }

# Wait up to $1 seconds for every live member of the snapshotted tree to go away.
drain() {
    local secs=$1 pids
    for _ in $(seq "$secs"); do
        mapfile -t pids < <(survivors)
        [ ${#pids[@]} -eq 0 ] && return 0
        sleep 1
    done
    return 1
}

# Signal a list of pids deepest-first, so a parent is still around to reap its children.
signal_tree() {
    local sig=$1; shift
    local pids=("$@") i
    for (( i=${#pids[@]}-1; i>=0; i-- )); do
        kill "-$sig" "${pids[i]}" 2>/dev/null
    done
}

zombies_before=$(count_zombies)
mapfile -t pids < <(collect_pids)

if [ ${#pids[@]} -eq 0 ]; then
    if port_busy; then
        echo "No sglang process found, but port ${PORT} is still listening."
        echo "Something else is holding it — check before relaunching."
        exit 1
    fi
    echo "No sglang server running."
    [ "$zombies_before" -gt 0 ] && echo "(${zombies_before} zombie entries present; see note at the end of this script.)"
    exit 0
fi

# Full tree, snapshotted now while the parent links still exist.
mapfile -t TREE < <(collect_tree)
echo "Server tree: ${#TREE[@]} processes (${TREE[*]})"

# --- phase 1: SIGTERM the launcher and let it take its own children down ----------
mapfile -t parents < <(collect_parents)
if [ ${#parents[@]} -gt 0 ]; then
    echo "Stopping sglang launcher (pids: ${parents[*]}) ..."
    kill -TERM "${parents[@]}" 2>/dev/null
    drain "$GRACE" && echo "Tree drained cleanly."
else
    echo "No launcher process — cleaning up orphaned children."
fi

# --- phase 2: anything still standing was orphaned or ignored the signal ----------
mapfile -t pids < <(survivors)
if [ ${#pids[@]} -gt 0 ]; then
    echo "Still up after launcher exit: ${pids[*]} — sending SIGTERM."
    signal_tree TERM "${pids[@]}"
    drain 10
fi

# --- phase 3: escalate ------------------------------------------------------------
mapfile -t pids < <(survivors)
if [ ${#pids[@]} -gt 0 ]; then
    echo "Still alive — sending SIGKILL to ${pids[*]}"
    signal_tree KILL "${pids[@]}"
    drain 5
fi

mapfile -t pids < <(survivors)
if [ ${#pids[@]} -gt 0 ]; then
    echo "FAILED to stop: ${pids[*]}"
    exit 1
fi

# Unmapping the weights takes a moment; the listener outlives the last process briefly.
for _ in $(seq 15); do
    port_busy || break
    sleep 1
done

if port_busy; then
    echo "Processes are gone but port ${PORT} is still listening."
    exit 1
fi

echo "Stopped. Port ${PORT} is free."

# --- zombie accounting -------------------------------------------------------------
# This script cannot reap zombies and does not pretend to. A zombie has already exited:
# it holds no memory, no GPU handle and no port, and signals do not apply to it. Its
# process-table entry is removed only when its PARENT calls wait(). Every orphan here is
# reparented to PID 1, which is `sleep infinity` -- it never reaps, so the entries stay
# until the container restarts. The permanent fix is host-side: add `--init` to the
# `docker run` in launch_docker.sh so tini runs as PID 1 and reaps orphans automatically
# (needs --recreate to take effect).
#
# Measured: parent-first shutdown took this from ~4 new zombies per stop to ~1. What it
# fixed was the sglang::scheduler / detokenizer leak -- those now exit while the launcher
# is still alive to wait() on them. What it cannot fix is a process that OUTLIVES its
# parent by design, notably multiprocessing.resource_tracker: it only exits once the
# launcher has closed its pipe, i.e. after the only process that could reap it is gone.
# That one is structural and needs --init. Zombie `bash` entries are usually not from the
# server at all -- they are the launching shell when the server was started with
# `nohup ... &` from a script that then exited.
zombies_after=$(count_zombies)
if [ "$zombies_after" -gt 0 ]; then
    new=$(( zombies_after - zombies_before ))
    echo
    echo "Zombies: ${zombies_after} total, ${new} new from this stop."
    ps -eo stat=,comm= 2>/dev/null | awk '$1 ~ /^Z/ {print $2}' | sort | uniq -c | sort -rn | sed 's/^/  /'
    echo "  Already-dead table entries: no memory, no GPU, no port held — harmless."
    echo "  They cannot be killed: PID 1 here is 'sleep infinity' and never reaps."
    echo "  Permanent fix: add --init to docker run in launch_docker.sh (host-side)."
fi
