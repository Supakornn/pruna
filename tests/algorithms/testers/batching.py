import pytest
from transformers import AutomaticSpeechRecognitionPipeline

from pruna import PrunaModel
from pruna.algorithms.batching.ifw import IFWBatcher
from pruna.algorithms.batching.ws2t import WhisperS2TWrapper, WS2TBatcher

from .base_tester import AlgorithmTesterBase


@pytest.mark.skip(reason="This test / the importing of whisper_s2t is affecting other tests.")
@pytest.mark.slow
class TestWhisperS2T(AlgorithmTesterBase):
    """Test the WhisperS2T batcher."""

    models = ["whisper_tiny"]
    reject_models = ["opt_125m"]
    allow_pickle_files = False
    algorithm_class = WS2TBatcher

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert isinstance(model.model, WhisperS2TWrapper)


class TestIFW(AlgorithmTesterBase):
    """Test the IFW batcher."""

    models = ["whisper_tiny"]
    reject_models = ["stable_diffusion_v1_4"]
    allow_pickle_files = False
    algorithm_class = IFWBatcher

    def post_smash_hook(self, model: PrunaModel) -> None:
        """Hook to modify the model after smashing."""
        assert isinstance(model.model, AutomaticSpeechRecognitionPipeline)
