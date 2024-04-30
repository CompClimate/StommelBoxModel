import torch.nn as nn


def reset_module_parameters(module: nn.Module):
    for layer in module.children():
        for m in layer.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
