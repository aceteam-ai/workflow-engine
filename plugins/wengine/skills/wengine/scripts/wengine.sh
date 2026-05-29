#!/bin/sh
#
# scripts/wengine.sh — self-contained launcher for the Workflow Engine CLI
# ---------------------------------------------
# Written for POSIX `sh` (not bash): use /bin/sh, dash, or bash in sh mode.
#
# If you run this script in place of typing "wengine", it will:
#   1) Make sure a small tool called "uv" exists (uv installs Python quickly).
#   2) Make sure a private Python environment exists under your home directory
#      with aceteam-workflow-engine installed.
#   3) Start the real wengine program inside that environment, forwarding argv.
#
# Nothing is installed into your current project directory; execution uses the
# private venv's Python only.
#
# Common flags use short portable forms (-p, -rf, -f) for BSD and busybox.
#
# Environment (optional):
#   WENGINE_RUNTIME_ROOT   — where bootstrap state lives (default ~/.wengine/runtime).
#   WENGINE_PYTHON         — interpreter for the venv (default 3.12; try 3.13, 3.14, …).
#   WENGINE_PACKAGE_SPEC   — pip requirement when not using editable (default pinned release).
#   WENGINE_EDITABLE_PATH  — absolute or relative path to a project root; if set, pip install -e
#                            that tree instead of WENGINE_PACKAGE_SPEC.
#
set -eu

# Version of the library we install from PyPI (must match plugin releases).
WENGINE_PINNED_VERSION="2.0.0rc12"

# What to install from PyPI when WENGINE_EDITABLE_PATH is unset.
WENGINE_PACKAGE_SPEC="${WENGINE_PACKAGE_SPEC:-aceteam-workflow-engine==${WENGINE_PINNED_VERSION}}"

WENGINE_PYTHON="${WENGINE_PYTHON:-3.12}"

# Resolved editable root (absolute path) when WENGINE_EDITABLE_PATH is set.
_wengine_ed_abs=""

if [ -n "${WENGINE_EDITABLE_PATH:-}" ]; then
	if [ ! -d "$WENGINE_EDITABLE_PATH" ]; then
		echo "wengine.sh: WENGINE_EDITABLE_PATH must be a directory: $WENGINE_EDITABLE_PATH" >&2
		exit 1
	fi
	_wengine_ed_abs="$(cd "$WENGINE_EDITABLE_PATH" && pwd)"
	WENGINE_INSTALL_ID="editable:${_wengine_ed_abs}"
else
	WENGINE_INSTALL_ID="$WENGINE_PACKAGE_SPEC"
fi

WENGINE_RUNTIME_ROOT="${WENGINE_RUNTIME_ROOT:-${HOME}/.wengine/runtime}"

VENV_DIR="${WENGINE_RUNTIME_ROOT}/venv"
SPEC_FILE="${WENGINE_RUNTIME_ROOT}/.install-spec"
UV_DIR="${WENGINE_RUNTIME_ROOT}/uv"
UV_EXE="${UV_DIR}/uv"

_wengine_lockdir=

_wengine_release_lock() {
	rm -rf "$_wengine_lockdir" 2>/dev/null || true
	_wengine_lockdir=
}

# If another bootstrap left .bootstrap-lock behind (crash, kill -9), clear it when
# the recorded PID is gone or missing.
_wengine_try_stale_lock_release() {
	if [ ! -d "$_wengine_lockdir" ]; then
		return 0
	fi
	if [ ! -f "$_wengine_lockdir/pid" ]; then
		rm -rf "$_wengine_lockdir" 2>/dev/null || true
		return 0
	fi
	_wengine_oldpid="$(cat "$_wengine_lockdir/pid" 2>/dev/null || true)"
	if [ -z "$_wengine_oldpid" ]; then
		rm -rf "$_wengine_lockdir" 2>/dev/null || true
		return 0
	fi
	if kill -0 "$_wengine_oldpid" 2>/dev/null; then
		return 0
	fi
	rm -rf "$_wengine_lockdir" 2>/dev/null || true
}

# --- uv (the installer tool) -----------------------------------------------
resolve_uv() {
	if command -v uv >/dev/null 2>&1; then
		UV_BIN="$(command -v uv)"
		return
	fi
	if [ ! -x "$UV_EXE" ]; then
		if ! command -v curl >/dev/null 2>&1; then
			echo "wengine.sh: need 'curl' on PATH to download uv, or install uv first (https://docs.astral.sh/uv/)." >&2
			exit 1
		fi
		mkdir -p "$UV_DIR"
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

venv_ok() {
	[ -f "$SPEC_FILE" ] || return 1
	[ "$(cat "$SPEC_FILE")" = "$WENGINE_INSTALL_ID" ] || return 1
	[ -x "${VENV_DIR}/bin/python" ] || return 1
	"${VENV_DIR}/bin/python" -c "import workflow_engine.cli.main" 2>/dev/null || return 1
}

install_venv() {
	_wengine_staging="venv.staging.$$"
	_wengine_staging_path="${WENGINE_RUNTIME_ROOT}/${_wengine_staging}"
	rm -rf "$_wengine_staging_path"

	"$UV_BIN" venv --python "$WENGINE_PYTHON" "$_wengine_staging_path"
	if [ -n "${WENGINE_EDITABLE_PATH:-}" ]; then
		"$UV_BIN" pip install --python "${_wengine_staging_path}/bin/python" -e "$_wengine_ed_abs"
	else
		"$UV_BIN" pip install --python "${_wengine_staging_path}/bin/python" "$WENGINE_PACKAGE_SPEC"
	fi

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

	_wengine_spec_tmp="${SPEC_FILE}.$$"
	printf '%s' "$WENGINE_INSTALL_ID" >"$_wengine_spec_tmp"
	mv -f "$_wengine_spec_tmp" "$SPEC_FILE"
}

bootstrap() {
	mkdir -p "$WENGINE_RUNTIME_ROOT"
	_wengine_lockdir="${WENGINE_RUNTIME_ROOT}/.bootstrap-lock"
	while true; do
		if mkdir "$_wengine_lockdir" 2>/dev/null; then
			printf '%s\n' $$ >"$_wengine_lockdir/pid"
			break
		fi
		_wengine_try_stale_lock_release
		if mkdir "$_wengine_lockdir" 2>/dev/null; then
			printf '%s\n' $$ >"$_wengine_lockdir/pid"
			break
		fi
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

exec "${VENV_DIR}/bin/python" -m workflow_engine.cli.main "$@"
