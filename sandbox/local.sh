#!/usr/bin/env bash
#
# Sandbox backend: run one command against the workspace, isolated with
# whatever the host OS actually provides.
#
#   local.sh <session-id> <timeout-seconds> <command...>
#   local.sh --probe                 # report the isolation available here
#
#   stdout / stderr  the command's own
#   exit code        the command's own, except:
#   exit 97          the sandbox itself failed (NOT a command failure)
#
# Exit 97 is load-bearing. A sandbox that could not start must never look like
# a test suite that failed, or the debugger agent spends the run chasing a bug
# that is not there.
#
# WHAT THIS ISOLATES
#
#   Everywhere      a clean environment built from an allowlist, so no host
#                   secret -- NVIDIA_API_KEY above all -- reaches repository
#                   code; the workspace as the working directory; a private
#                   HOME and TMPDIR per session; a file-size ceiling.
#   Linux / WSL     additionally a network namespace of its own via
#                   `unshare -rn`, and a process-count ceiling.
#   Windows         nothing beyond the above. There is no namespace, no
#                   filesystem boundary, and no way to stop the command
#                   reaching the network or the rest of the disk.
#
# So on Windows this is NOT a security boundary: the agent's code runs as you.
# It is appropriate for a repository you would have run `pytest` in anyway.
# For a real boundary use the Docker backend (SE_AGENT_SANDBOX=docker), or run
# this script inside WSL where the namespace primitives exist.

set -u

SANDBOX_FAILED=97

die() { echo "sandbox: $*" >&2; exit "$SANDBOX_FAILED"; }

# A Windows path contains a colon, which is also the PATH separator, so
# "C:/x" prepended to PATH silently becomes the two nonexistent entries "C"
# and "/x". Everything that ends up in PATH goes through here first.
to_posix() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -u "$1"; else printf '%s' "$1"; fi
}

# --------------------------------------------------------------- probe mode

# What the caller reports in preflight, so nobody believes they have
# containment they do not have.
probe() {
  local features="clean-env workspace-cwd file-size-limit"
  if command -v unshare >/dev/null 2>&1 && unshare -rn true 2>/dev/null; then
    features="$features network-namespace"
  fi
  if ulimit -u >/dev/null 2>&1; then
    features="$features process-limit"
  fi
  case "$(uname -s 2>/dev/null || echo unknown)" in
    Linux)            echo "linux: $features" ;;
    Darwin)           echo "macos: $features" ;;
    MINGW*|MSYS*|CYG*) echo "windows: $features (no isolation boundary)" ;;
    *)                echo "unknown: $features" ;;
  esac
}

[ "${1:-}" = "--probe" ] && { probe; exit 0; }

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
for d in "$workspace/.venv/bin" "$workspace/.venv/Scripts"          "$workspace/venv/bin" "$workspace/venv/Scripts"; do
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

runner=(sh -c "$command")

# `unshare -rn` maps the caller to root in a new user namespace and gives the
# command a network namespace with only a down loopback -- no privileges
# required, and no route to anything. Absent on Windows and older kernels, so
# probe before committing to it.
if [ "${SE_AGENT_SANDBOX_NETWORK:-0}" != "1" ]; then
  if command -v unshare >/dev/null 2>&1 && unshare -rn true 2>/dev/null; then
    runner=(unshare -rn "${runner[@]}")
  fi
fi

# A wall-clock ceiling, where coreutils provides one. The caller enforces its
# own timeout as well and kills the whole process group, which is what
# actually guarantees termination; this is the polite first attempt.
if command -v timeout >/dev/null 2>&1; then
  runner=(timeout --kill-after=10s "${timeout_s}s" "${runner[@]}")
fi

# ------------------------------------------------------------------ execute

# `env -i` is the guarantee that matters and the reason this is a script you
# can read: the command starts from nothing and receives only what is listed
# here. Host credentials are not omitted by convention, they are absent.
exec env -i \
  PATH="${PATH}" \
  HOME="$scratch" \
  TMPDIR="$scratch" \
  TEMP="$scratch" \
  TMP="$scratch" \
  LANG="${LANG:-C.UTF-8}" \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  CI=1 \
  "${runner[@]}"
