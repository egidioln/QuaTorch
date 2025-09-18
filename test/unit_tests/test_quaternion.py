import math

import pytest
import torch

from quatorch import Quaternion


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


def test_to_from_rotation_matrix():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    R = q.to_rotation_matrix()
    expected_R = torch.eye(3)
    assert torch.allclose(R, expected_R)

    q = Quaternion(0.7071, 0.7071, 0.0, 0.0)  # 90 degrees around x-axis
    R = q.to_rotation_matrix()
    expected_R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(R, expected_R, atol=1e-4)

    q_from_R = Quaternion.from_rotation_matrix(R)
    assert torch.allclose(q_from_R.normalize(), q.normalize(), atol=1e-4)


def test_from_axis_angle_and_rotate_vector():
    # 90 degree rotation around z-axis
    axis = torch.tensor([0.0, 0.0, 1.0])
    angle = torch.tensor(math.pi / 2)
    q = Quaternion.from_axis_angle(axis, angle)

    # Rotate unit x vector to unit y vector
    v = torch.tensor([1.0, 0.0, 0.0])
    v_rot = q.rotate_vector(v)
    expected = torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(v_rot, expected, atol=1e-5)

    # Axis-angle round trip
    axis_out, angle_out = q.to_axis_angle()
    assert torch.allclose(angle_out, angle, atol=1e-5)
    # axis may differ in sign when angle is near pi, compare absolute direction
    assert torch.allclose(axis_out.abs(), axis.abs(), atol=1e-5)


def test_slerp_and_pow():
    # interpolate between identity and 180deg rotation about x
    q0 = Quaternion(1.0, 0.0, 0.0, 0.0).normalize()
    q1 = Quaternion.from_axis_angle(
        torch.tensor([1.0, 0.0, 0.0]), torch.tensor(math.pi)
    )

    # halfway slerp should be a 90deg rotation about x
    q_half = q0.slerp(q1, 0.5)
    axis, angle = q_half.to_axis_angle()
    assert torch.allclose(angle, torch.tensor(math.pi / 2), atol=1e-5)

    # pow: raising q1 to 0.5 should equal q_half (since q0 is identity)
    q1_sqrt = q1.pow(0.5)
    axis2, angle2 = q1_sqrt.to_axis_angle()
    assert torch.allclose(angle2, torch.tensor(math.pi / 2), atol=1e-5)


def test_log_exp():
    # Test log and exp round trip for a non-trivial quaternion
    q = Quaternion.from_axis_angle(
        torch.tensor([0.0, 1.0, 0.0]), torch.tensor(math.pi / 3)
    )
    q = q.normalize()

    log_q = torch.log(q)
    # log should have zero scalar part
    assert torch.allclose(log_q[..., 0], torch.tensor(0.0), atol=1e-6)

    q_rec = Quaternion(torch.exp(log_q))
    # exp(log(q)) may produce a quaternion proportional to q; normalize before compare
    assert torch.allclose(q_rec.normalize(), q.normalize(), atol=1e-5)
