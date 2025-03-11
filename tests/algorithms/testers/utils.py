from typing import Any


def get_model_sparsity(model: Any) -> float:
    """Get the sparsity of the model."""
    total_params = 0
    zero_params = 0
    for module in model.modules():
        if hasattr(module, "weight"):
            # Use the effective weight if pruning has been applied.
            weight = module.weight_orig * module.weight_mask if hasattr(module, "weight_mask") else module.weight
            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()
    return zero_params / total_params if total_params > 0 else 0.0
