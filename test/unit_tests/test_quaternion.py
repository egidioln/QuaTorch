import pytest


from quatorch import Quaternion
import torch
import math


def test_quaternion_init():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    assert torch.allclose(q, torch.tensor([1.0, 0.0, 0.0, 0.0]))

    w = torch.tensor([1.0, 0.0])
    x = torch.tensor([0.0, 1.0])
    y = torch.tensor([0.0, 2.0])
    z = torch.tensor([0.0, 3.0])
    q = Quaternion(w, x, y, z)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    assert torch.allclose(q, expected)

    with pytest.raises(ValueError):
        Quaternion(1.0, 2.0, 3.0)  # Missing one component

    with pytest.raises(ValueError):
        Quaternion(torch.tensor([1.0, 2.0]), torch.tensor([3.0]))  # Shape mismatch
    with pytest.raises(ValueError):
        Quaternion(
            torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0])
        )  # Shape mismatch
    with pytest.raises(ValueError):
        Quaternion(
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([5.0, 6.0]),
            torch.tensor([7.0]),
        )  # Shape mismatch
    with pytest.raises(ValueError):
        Quaternion(
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([5.0, 6.0]),
            torch.tensor([7.0, 8.0, 9.0]),
        )  # Shape mismatch
    with pytest.raises(ValueError):
        Quaternion(
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([5.0, 6.0]),
            torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
        )  # Shape mismatch
    with pytest.raises(ValueError):
        Quaternion(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([5.0, 6.0]),
            torch.tensor([7.0, 8.0]),
            torch.tensor([9.0, 10.0]),
        )  # Shape mismatch


def test_quaternion_operations():
    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(0.5, 1.0, 1.5, 2.0)

    # Addition
    q_sum = q1 + q2
    expected_sum = torch.tensor([1.5, 3.0, 4.5, 6.0])
    assert torch.allclose(q_sum, expected_sum)

    # Subtraction
    q_diff = q1 - q2
    expected_diff = torch.tensor([0.5, 1.0, 1.5, 2.0])
    assert torch.allclose(q_diff, expected_diff)

    # Multiplication
    q_prod = q1 * q2
    expected_prod = torch.tensor([-14.0, 2.0, 3.0, 4.0])
    assert torch.allclose(q_prod, expected_prod)

    # Division
    q_div = q1 / q2
    expected_div = torch.tensor([2.0, 0.0, 0.0, 0.0])
    assert torch.allclose(q_div, expected_div)

    # Conjugate
    q_conj = q1.conjugate()
    expected_conj = torch.tensor([1.0, -2.0, -3.0, -4.0])
    assert torch.allclose(q_conj, expected_conj)

    # Norm
    q_norm = q1.norm()
    expected_norm = math.sqrt(30)
    assert torch.allclose(q_norm, torch.tensor(expected_norm))

    # Inverse
    q_inv = q1.inverse()
    expected_inv = torch.tensor([1 / 30, -2 / 30, -3 / 30, -4 / 30])
    assert torch.allclose(q_inv, expected_inv)

    # Normalize
    q_normalized = q1.normalize()
    expected_normalized = torch.tensor(
        [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)]
    )
    assert torch.allclose(q_normalized, expected_normalized)
    assert torch.allclose(q_normalized.norm(), torch.tensor(1.0))
