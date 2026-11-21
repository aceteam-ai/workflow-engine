"""Regression coverage for the project-wide pytest timeout configuration."""

import subprocess
import sys
from pathlib import Path


def test_project_timeout_settings_are_loaded() -> None:
    """Default pytest discovery loads the project-wide timeout settings."""
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "tests/test_pytest_timeout.py",
        ],
        capture_output=True,
        check=False,
        cwd=project_root,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0
    output = completed.stdout + completed.stderr
    assert "configfile: pytest.ini" in output
    assert "timeout: 180.0s" in output
    assert "timeout method: signal" in output


def test_timeout_plugin_interrupts_a_hanging_test(tmp_path: Path) -> None:
    """A global timeout setting interrupts a hang in an isolated pytest process."""
    hanging_test = tmp_path / "test_hanging.py"
    hanging_test.write_text(
        "import time\n\ndef test_hangs_forever():\n    time.sleep(60)\n"
    )

    timeout_config = tmp_path / "pytest.ini"
    timeout_config.write_text(
        "[pytest]\naddopts = --timeout=0.2 --timeout-method=signal\n"
    )

    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            str(timeout_config),
            str(hanging_test),
        ],
        capture_output=True,
        check=False,
        cwd=project_root,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 1
    assert "Timeout" in completed.stdout + completed.stderr
