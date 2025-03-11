from typing import Any

import pytest

from pruna import SmashConfig
from pruna.engine.pruna_model import PrunaModel
from pruna.evaluation.metrics.metric_model_architecture import ModelArchitectureMetric


@pytest.mark.parametrize(
    "model_fixture, device",
    [
        pytest.param("noref_resnet_18", "cpu", marks=pytest.mark.cpu),
        pytest.param("stable_diffusion_v1_4", "cuda", marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
)
def test_model_architecture_metrics(model_fixture: tuple[Any, SmashConfig], device: str) -> None:
    """Test the model architecture metrics."""
    model, smash_config = model_fixture
    metric = ModelArchitectureMetric(device=device)
    pruna_model = PrunaModel(model, smash_config=smash_config)
    results = metric.compute(pruna_model, smash_config.test_dataloader())
    assert results["total_macs"] > 0
    assert results["total_params"] > 0
