from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pdm_bench.pipelines.dl.config import DLPipelineConfig


HYDRA_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
CWRU_ROOT_SUFFIX = Path("datasets/javadseraj-cwru-bearing-fault-data-set/Datasets/CWRU")
PU_ROOT_SUFFIX = Path("datasets/paderborn-university-bearing-dataset")


def _discover_task_presets() -> list[str]:
    task_dir = HYDRA_CONFIG_DIR / "task"
    discovered: list[str] = []
    for path in sorted(task_dir.rglob("*.yaml")):
        option = path.relative_to(task_dir).with_suffix("").as_posix()
        if option == "none":
            continue
        discovered.append(option)
    return discovered


ALL_TASK_PRESETS = _discover_task_presets()


def test_all_task_presets_compose_for_valid_supervised_dl_runs(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / CWRU_ROOT_SUFFIX).mkdir(parents=True, exist_ok=True)
    (workspace / PU_ROOT_SUFFIX).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WORKSPACE", str(workspace))

    for task in ALL_TASK_PRESETS:
        with initialize_config_dir(version_base="1.3", config_dir=str(HYDRA_CONFIG_DIR)):
            cfg = compose(config_name="config", overrides=[f"task={task}"])

        data = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(data, dict)
        data.pop("hydra", None)

        parsed = DLPipelineConfig.from_dict(data)
        assert parsed.run.name
        assert parsed.models
        assert parsed.windowing.size > 0
        assert parsed.train.epochs > 0
        assert parsed.dataset.loader in {"cwru", "pu"}
