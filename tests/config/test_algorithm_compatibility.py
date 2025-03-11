import pytest

from pruna.algorithms import PRUNA_ALGORITHMS


@pytest.mark.cpu
def test_algorithm_compatibility_consistency() -> None:
    """Test the consistency of algorithm compatibility."""
    for algorithm_group_name, algorithm_group in PRUNA_ALGORITHMS.items():
        for algorithm_name, algorithm_class in algorithm_group.items():
            for group, algs in algorithm_class.compatible_algorithms.items():
                for alg in algs:
                    compatible_alg = PRUNA_ALGORITHMS[group][alg]
                    if (
                        algorithm_group_name not in compatible_alg.compatible_algorithms.keys()
                        or algorithm_name not in compatible_alg.compatible_algorithms[algorithm_group_name]
                    ):
                        raise ValueError(f"Inconsistent algorithm compatibility: {algorithm_name} and {alg}")
