from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pdm_tools.main.pipelines.common.io_utils import load_env_file, prepare_run_dir
from pdm_tools.main.pipelines.dl.config import RunSpec


if TYPE_CHECKING:
    from pathlib import Path


def test_prepare_run_dir_creates_folder(tmp_path: Path):
    run = RunSpec(name="my run", output_dir=str(tmp_path))
    run_dir, _timestamp = prepare_run_dir(run)

    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path
    assert "my-run" in run_dir.name


def test_load_env_file_sets_env(monkeypatch, tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("FOO=bar\nexport BAZ=qux\n")

    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.setenv("BAZ", "keep")

    load_env_file(env_path)

    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "keep"
