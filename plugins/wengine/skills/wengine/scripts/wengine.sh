#!/bin/sh
#
# scripts/wengine.sh — self-contained launcher for the Workflow Engine CLI
# ---------------------------------------------
# Written for POSIX `sh` (not bash): use /bin/sh, dash, or bash in sh mode.
#
# If you run this script in place of typing "wengine", it will:
#   1) Make sure a small tool called "uv" exists (uv installs Python packages quickly).
#   2) Make sure a private Python environment exists under your home directory,
#      and that it contains the "aceteam-workflow-engine" package.
#   3) Start the real wengine program inside that environment, passing along
#      any arguments you gave (e.g. "node list", "--help").
#
# Nothing is installed into the project folder you are in, and we do not rely
# on whatever Python might be installed system-wide. After setup, we hand off
# to that private environment's Python and do not stay in this shell script.
#
# Common flags use short portable forms (-p, -rf, -f) so they work on BSD and busybox.
#
# Advanced: point at a local checkout instead of PyPI:
#   WENGINE_PACKAGE_SPEC='-e /path/to/workflow-engine' sh scripts/wengine.sh --help
#
set -eu

# Version of the library we install from PyPI (must match plugin releases).
WENGINE_PINNED_VERSION="2.0.0rc11"

# What to "pip install" into the private environment (override for local dev).
WENGINE_PACKAGE_SPEC="${WENGINE_PACKAGE_SPEC:-aceteam-workflow-engine==${WENGINE_PINNED_VERSION}}"

# All managed files live here (override to put the runtime somewhere else).
WENGINE_RUNTIME_ROOT="${WENGINE_RUNTIME_ROOT:-${HOME}/.wengine/runtime}"

# Short names for important paths under that root.
VENV_DIR="${WENGINE_RUNTIME_ROOT}/venv"
SPEC_FILE="${WENGINE_RUNTIME_ROOT}/.install-spec"
UV_DIR="${WENGINE_RUNTIME_ROOT}/uv"
UV_EXE="${UV_DIR}/uv"

# Set by bootstrap() so the trap handler can remove the lock without nested functions
# (nested functions are not POSIX).
_wengine_lockdir=

_wengine_release_lock() {
	rmdir "$_wengine_lockdir" 2>/dev/null || true
	_wengine_lockdir=
}

# --- uv (the installer tool) -----------------------------------------------
# We need the `uv` program to create the Python environment and install packages.
# Prefer an existing `uv` on the machine; if there is none, download one into
# UV_DIR once and reuse it forever after.
resolve_uv() {
	if command -v uv >/dev/null 2>&1; then
		UV_BIN="$(command -v uv)"
		return
	fi
	if [ ! -x "$UV_EXE" ]; then
		mkdir -p "$UV_DIR"
		# Fail on HTTP errors, stay quiet except real errors, follow redirects.
		curl --fail --silent --show-error --location \
			https://astral.sh/uv/install.sh |
			env UV_INSTALL_DIR="$UV_DIR" sh
	fi
	if [ ! -x "$UV_EXE" ]; then
		echo "wengine.sh: failed to install uv to $UV_EXE" >&2
		exit 1
	fi
	UV_BIN="$UV_EXE"
}

# --- Is the existing environment still good? --------------------------------
# We record what we installed in SPEC_FILE. If that matches what we want, the
# venv folder exists, and Python can import the CLI, we skip reinstalling.
venv_ok() {
	[ -f "$SPEC_FILE" ] || return 1
	[ "$(cat "$SPEC_FILE")" = "$WENGINE_PACKAGE_SPEC" ] || return 1
	[ -x "${VENV_DIR}/bin/python" ] || return 1
	"${VENV_DIR}/bin/python" -c "import workflow_engine.cli.main" 2>/dev/null || return 1
}

# --- Build a fresh private Python environment -------------------------------
# Work entirely in a temporary folder until everything succeeds, then "swap"
# it into place so you never end up with half an install as the main venv.
# (No `local` — not POSIX; these names are only used inside this function.)
install_venv() {
	_wengine_staging="venv.staging.$$"
	_wengine_staging_path="${WENGINE_RUNTIME_ROOT}/${_wengine_staging}"
	rm -rf "$_wengine_staging_path"

	# Create an isolated Python 3.12 environment (does not touch system Python).
	"$UV_BIN" venv --python 3.12 "$_wengine_staging_path"
	# Install the workflow engine and its dependencies into that environment only.
	"$UV_BIN" pip install --python "${_wengine_staging_path}/bin/python" "$WENGINE_PACKAGE_SPEC"

	# Swap in the new venv without deleting the old one first:
	# rename current -> backup, rename staging -> current, then delete backup.
	# If the final rename fails, put the old venv back so you still have a working CLI.
	_wengine_old="${VENV_DIR}.old.$$"
	rm -rf "$_wengine_old"
	if [ -d "$VENV_DIR" ]; then
		mv "$VENV_DIR" "$_wengine_old"
	fi
	if ! mv "$_wengine_staging_path" "$VENV_DIR"; then
		if [ -d "$_wengine_old" ]; then
			mv "$_wengine_old" "$VENV_DIR"
		fi
		echo "wengine.sh: failed to promote new venv" >&2
		return 1
	fi
	rm -rf "$_wengine_old"

	# Remember what we installed, using a temp file + rename so the file is
	# never half-written if something crashes mid-save.
	_wengine_spec_tmp="${SPEC_FILE}.$$"
	printf '%s' "$WENGINE_PACKAGE_SPEC" >"$_wengine_spec_tmp"
	mv -f "$_wengine_spec_tmp" "$SPEC_FILE"
}

# --- Run setup if needed (with a simple lock) --------------------------------
# Two terminals might run this at once. Only one should install at a time: we
# create a tiny lock directory (creating a directory is atomic on local disks).
# Others wait in a loop until the lock disappears.
#
# `trap` tells the shell: when any of these happen, run `_wengine_release_lock` once.
#   EXIT: This process is exiting (success or failure). Drops the lock if
#         we bail out before the normal cleanup path.
#   INT:  Ctrl+C (interrupt).
#   HUP:  Hangup (e.g. terminal closed).
#   TERM: Polite shutdown (e.g. `kill`).
bootstrap() {
	mkdir -p "$WENGINE_RUNTIME_ROOT"
	_wengine_lockdir="${WENGINE_RUNTIME_ROOT}/.bootstrap-lock"
	# sleep accepts fractions on many systems; POSIX only requires integer seconds.
	while ! mkdir "$_wengine_lockdir" 2>/dev/null; do
		sleep 0.2 2>/dev/null || sleep 1
	done
	trap '_wengine_release_lock' EXIT INT HUP TERM
	resolve_uv
	if ! venv_ok; then
		install_venv
	fi
	_wengine_release_lock
	trap - EXIT INT HUP TERM
}

bootstrap

# Replace this shell process with the real CLI, forwarding all arguments.
exec "${VENV_DIR}/bin/python" -m workflow_engine.cli.main "$@"
