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

from abc import abstractmethod
from typing import Any, Dict

from pruna.evaluation.metrics.metric_base import BaseMetric


class StatefulMetric(BaseMetric):
    """
    Base class for all metrics that have state functionality.

    A stateful metric maintains internal state variables that accumulate information
    across multiple batches or iterations. Unlike simple metrics that compute values
    independently for each input, stateful metrics track running statistics or
    aggregated values over time.
    """

    def __init__(self) -> None:
        """Initialize the StatefulMetric class."""
        super().__init__()
        self.metric_config: Dict[str, Any] = {}
        self.metric_name: str = ""
        self.call_type: str = ""

    def add_state(self, *args, **kwargs) -> None:
        """
        Add state variables to the metric.

        Parameters
        ----------
        *args : Any
            The arguments to pass to the metric.
        **kwargs : Any
            The keyword arguments to pass to the metric.
        """
        pass

    def forward(self, *args, **kwargs) -> None:
        """
        Compute the metric value.

        Parameters
        ----------
        *args : Any
            The arguments to pass to the metric.
        **kwargs : Any
            The keyword arguments to pass to the metric.
        """
        pass

    def reset(self) -> None:
        """Reset the metric state."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Override this method to update the state variables of your metric.

        Parameters
        ----------
        *args : Any
            The arguments to pass to the metric.
        **kwargs : Any
            The keyword arguments to pass to the metric.
        """

    @abstractmethod
    def compute(self) -> Any:
        """Override this method to compute the final metric value."""

    def is_pairwise(self) -> bool:
        """
        Check if a metric is pairwise.

        Returns
        -------
        bool
            True if the metric is pairwise, False otherwise.
        """
        return self.call_type.startswith("pairwise")
