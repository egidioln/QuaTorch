import torch

from quatorch.quaternion import Quaternion

assert torch.all(
    Quaternion(1.0, 0.0, 0.0, 0.0) + Quaternion(0.0, 1.0, 0.0, 0.0)
    == Quaternion(1.0, 1.0, 0.0, 0.0)
)
