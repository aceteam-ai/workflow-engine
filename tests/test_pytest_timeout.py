"""Regression coverage for the project-wide pytest timeout configuration."""

import subprocess
import sys
from pathlib import Path


def test_timeout_plugin_interrupts_a_hanging_test(tmp_path: Path) -> None:
    """Run the hang in an isolated pytest process so this suite cannot wedge."""
    hanging_test = tmp_path / "test_hanging.py"
    hanging_test.write_text(
        "import time\n"
        "\n"
        "import pytest\n"
        "\n"
        "\n"
        "@pytest.mark.timeout(0.2)\n"
        "def test_hangs_forever():\n"
        "    time.sleep(60)\n"
    )

    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            str(project_root / "pyproject.toml"),
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
