import tempfile
from typing import Any

import pytest

from pruna import PrunaModel, SmashConfig
from pruna.evaluation.metrics.metric_memory import GPUMemoryMetric


@pytest.mark.parametrize(
    "model_fixture, mode",
    [
        pytest.param("stable_diffusion_v1_4", "disk", marks=pytest.mark.cuda),
        pytest.param("resnet_18", "disk", marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
)
def test_memory_metrics(model_fixture: tuple[Any, SmashConfig], mode: str) -> None:
    """Test the memory metrics."""
    model, smash_config = model_fixture
    with tempfile.TemporaryDirectory() as temp_dir:
        smash_config_dummy = SmashConfig()
        model = PrunaModel(model, smash_config=smash_config_dummy)
        model.save_pretrained(temp_dir)

        metric = GPUMemoryMetric(mode=mode)
        result = metric.compute(model, dataloader=smash_config.test_dataloader())
        assert result[f"{mode}_memory"] > 0
