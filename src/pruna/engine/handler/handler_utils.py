# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from pruna.engine.handler.handler_diffuser import DiffuserHandler
from pruna.engine.handler.handler_inference import InferenceHandler
from pruna.engine.handler.handler_standard import StandardHandler
from pruna.engine.handler.handler_transformer import TransformerHandler


def register_inference_handler(model: Any) -> InferenceHandler:
    """
    Register an inference handler for the model. The handler is chosen based on the model type.

    Parameters
    ----------
    model : Any
        The model to register a handler for.

    Returns
    -------
    InferenceHandler
        The registered handler.
    """
    if "diffusers" in model.__module__:
        return DiffuserHandler()
    elif "transformers" in model.__module__:
        return TransformerHandler()
    else:
        return StandardHandler()
