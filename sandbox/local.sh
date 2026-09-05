#!/usr/bin/env bash
#
# The sandbox. Runs one command against the workspace, isolated with the
# strongest mechanism this host actually provides.
#
#   local.sh <session-id> <timeout-seconds> <command...>
#   local.sh --probe                 # report the tier available here
#   local.sh --tier                  # just the tier name
#
#   stdout / stderr  the command's own
#   exit code        the command's own, except:
#   exit 97          the sandbox itself failed (NOT a command failure)
#
# Exit 97 is load-bearing. A sandbox that could not start must never look like
# a test suite that failed, or the debugger agent spends the run chasing a bug
# that is not there.
#
# ISOLATION TIERS, strongest first. Each is probed by *running* it, not by
# looking for the binary, so a mechanism that is present but broken (a kernel
# with unprivileged user namespaces disabled) demotes instead of failing
# every run.
#
#   bwrap     A real boundary. Fresh mount namespace: the workspace is the
#             only writable path and most of the filesystem is not visible at
#             all. No network, no new privileges, its own PID and IPC space.
#   unshare   Network and user namespace only. The filesystem is still the
#             host's, so this bounds what the command can *reach*, not what
#             it can read.
#   none      A clean environment and a working directory. NOT a boundary.
#
# Every tier gets the environment allowlist: the command starts from `env -i`
# and receives only what is listed at the bottom of this file, so
# NVIDIA_API_KEY and other host secrets are absent rather than unmentioned.
#
# On Windows only `none` is reachable, because the namespace primitives are
# Linux's. Set SE_AGENT_SANDBOX_WSL=1 to run inside WSL and reach the others.

set -u

SANDBOX_FAILED=97

die() { echo "sandbox: $*" >&2; exit "$SANDBOX_FAILED"; }

# A Windows path contains a colon, which is also the PATH separator, so
# "C:/x" prepended to PATH silently becomes the two nonexistent entries "C"
# and "/x". Everything that ends up in PATH goes through here first.
to_posix() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -u "$1"; else printf '%s' "$1"; fi
}

# --------------------------------------------------------------------- tiers

bwrap_works() {
  command -v bwrap >/dev/null 2>&1 || return 1
  bwrap --unshare-all --ro-bind / / --die-with-parent true >/dev/null 2>&1
}

unshare_works() {
  command -v unshare >/dev/null 2>&1 || return 1
  unshare -rn true >/dev/null 2>&1
}

tier() {
  if bwrap_works; then echo bwrap
  elif unshare_works; then echo unshare
  else echo none
  fi
}

probe() {
  case "$(tier)" in
    bwrap)   echo "bwrap: filesystem+network+pid namespaces, workspace-only writes" ;;
    unshare) echo "unshare: network namespace, no filesystem boundary" ;;
    none)    echo "none: clean environment only -- NOT an isolation boundary" ;;
  esac
}

case "${1:-}" in
  --probe) probe; exit 0 ;;
  --tier)  tier;  exit 0 ;;
esac

# ------------------------------------------------------------------ arguments

[ "$#" -ge 3 ] || die "usage: local.sh <session-id> <timeout-seconds> <command...>"

session="$1"
timeout_s="$2"
shift 2
command="$*"

[ -n "$command" ] || die "empty command"

workspace="${SE_AGENT_WORKSPACE:-}"
[ -n "$workspace" ] || die "SE_AGENT_WORKSPACE is not set"
[ -d "$workspace" ] || die "workspace is not a directory: $workspace"
cd "$workspace" || die "cannot enter workspace: $workspace"
workspace="$(pwd -P)"

# Per-session scratch, so concurrent runs cannot see each other's temp files
# and nothing the command writes lands in the real HOME.
scratch="${TMPDIR:-/tmp}/se-agent-$session"
mkdir -p "$scratch" || die "cannot create scratch directory: $scratch"
trap 'rm -rf "$scratch"' EXIT

# ------------------------------------------------------------- interpreter

# A clean environment also means no active virtualenv, so the command would
# otherwise get whichever Python happens to be first on the system PATH --
# usually one without the project's test dependencies. Prefer the workspace's
# own environment, and let the caller override outright.
extra=""
for d in "$workspace/.venv/bin" "$workspace/.venv/Scripts" \
         "$workspace/venv/bin" "$workspace/venv/Scripts"; do
  if [ -d "$d" ]; then extra="$(to_posix "$d")"; break; fi
done
if [ -n "${SE_AGENT_SANDBOX_PATH:-}" ]; then
  extra="$(to_posix "$SE_AGENT_SANDBOX_PATH")${extra:+:$extra}"
fi
[ -n "$extra" ] && PATH="$extra:$PATH"

# ------------------------------------------------------------------ limits

# No output file larger than 512MB, and no fork bomb where the shell can stop
# one. `ulimit -v` is deliberately not set: capping address space reliably
# kills CPython before it reaches main().
ulimit -f 524288 2>/dev/null || true
ulimit -u 512 2>/dev/null || true

# ---------------------------------------------------------------- isolation

want_network="${SE_AGENT_SANDBOX_NETWORK:-0}"
active="$(tier)"

# An explicit floor. Refuse to run below the tier the caller demands rather
# than quietly giving them less isolation than they asked for.
case "${SE_AGENT_SANDBOX_REQUIRE:-none}" in
  bwrap)
    [ "$active" = "bwrap" ] || die "SE_AGENT_SANDBOX_REQUIRE=bwrap, best available is '$active'"
    ;;
  unshare)
    [ "$active" = "none" ] && die "SE_AGENT_SANDBOX_REQUIRE=unshare, best available is 'none'"
    ;;
  none) ;;
  *) die "unknown SE_AGENT_SANDBOX_REQUIRE: ${SE_AGENT_SANDBOX_REQUIRE:-}" ;;
esac

runner=(sh -c "$command")

case "$active" in
  bwrap)
    bw=(bwrap --die-with-parent --new-session --unshare-all)
    [ "$want_network" = "1" ] && bw+=(--share-net)
    # A minimal read-only system, then the workspace as the one writable path.
    for d in /usr /bin /sbin /lib /lib32 /lib64 /etc /opt; do
      [ -e "$d" ] && bw+=(--ro-bind "$d" "$d")
    done
    bw+=(--proc /proc --dev /dev --tmpfs /run --tmpfs /tmp)
    bw+=(--bind "$workspace" "$workspace")
    bw+=(--bind "$scratch" "$scratch")
    bw+=(--chdir "$workspace")
    runner=("${bw[@]}" "${runner[@]}")
    ;;
  unshare)
    # `unshare -rn` maps the caller to root in a new user namespace and gives
    # the command a network namespace with only a down loopback -- no
    # privileges required, and no route to anything.
    [ "$want_network" = "1" ] || runner=(unshare -rn "${runner[@]}")
    ;;
esac

# A wall-clock ceiling, where coreutils provides one. The caller enforces its
# own timeout as well and kills the whole process group, which is what
# actually guarantees termination; this is the polite first attempt.
if command -v timeout >/dev/null 2>&1; then
  runner=(timeout --kill-after=10s "${timeout_s}s" "${runner[@]}")
fi

# ------------------------------------------------------------------ execute

# `env -i` is the guarantee that holds at every tier, and the reason this is a
# script you can read: the command starts from nothing and receives only what
# is listed here. Host credentials are not omitted by convention, they are
# absent.
allow=(
  PATH="$PATH"
  HOME="$scratch"
  TMPDIR="$scratch"
  TEMP="$scratch"
  TMP="$scratch"
  LANG="${LANG:-C.UTF-8}"
  PYTHONUNBUFFERED=1
  PYTHONDONTWRITEBYTECODE=1
  PIP_DISABLE_PIP_VERSION_CHECK=1
  CI=1
)

# Windows needs these to start a process at all -- CPython cannot open a
# socket without SystemRoot, so omitting them would break the sandbox rather
# than harden it. Neither carries a credential.
[ -n "${SYSTEMROOT:-}" ] && allow+=(SYSTEMROOT="$SYSTEMROOT")
[ -n "${COMSPEC:-}" ] && allow+=(COMSPEC="$COMSPEC")

exec env -i "${allow[@]}" "${runner[@]}"
