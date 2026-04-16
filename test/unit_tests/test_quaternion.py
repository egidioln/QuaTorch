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

    q = Quaternion(x=x, y=y, z=z, w=w)
    assert torch.allclose(q, expected)

    assert torch.allclose(q * 2, expected * 2)

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
    assert torch.allclose(4 * q1, q1 * 4)

    # Broadcasted multiplication
    tq1 = Quaternion(torch.randn(3, 4))
    multiplying_tensor = torch.randn(2, 1, 1)
    result = multiplying_tensor * tq1
    assert result.shape == (2, 3, 4)

    # Division
    q_div = q1 / q2
    expected_div = torch.tensor([2.0, 0.0, 0.0, 0.0])
    assert torch.allclose(q_div, expected_div)

    assert torch.allclose(q1.inverse(), 1 / q1)
    assert torch.allclose(q2 * q1.inverse(), q2 / q1)

    # Broadcasted division
    tq1 = Quaternion(torch.randn(3, 4))
    dividend_tensor = torch.randn(2, 1, 1)
    result = dividend_tensor / tq1
    assert result.shape == (2, 3, 4)

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


def test_rotate_vector_broadcasts_as_it_should():
    q = Quaternion.from_axis_angle(
        torch.tensor([0.0, 1.0, 0.0]), torch.tensor(math.pi / 2)
    ).normalize()

    v = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    v_rot = q.rotate_vector(v)
    expected = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert torch.allclose(v_rot, expected, atol=1e-5)

    # Test batch of quaternions
    q_batch = Quaternion.from_axis_angle(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([math.pi / 2, math.pi]),
    ).normalize()

    v = torch.tensor([1.0, 0.0, 0.0])
    v_rot_batch = q_batch.rotate_vector(v)
    expected_batch = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert torch.allclose(v_rot_batch, expected_batch, atol=1e-5)


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

    # pow: raising q1 to 0.5 should equal q_half (since q0 is identity)
    q1_sqrt = q1.pow(torch.tensor(0.5))
    axis2, angle2 = q1_sqrt.to_axis_angle()
    assert torch.allclose(angle2, torch.tensor(math.pi / 2), atol=1e-5)


def test_slerp_batch():
    t = torch.linspace(0, 1, steps=6).unsqueeze(-1)
    q = Quaternion.from_axis_angle(
        torch.tensor([0.0, 1.0, 0.0]), torch.tensor(math.pi / 3)
    )
    q = q.normalize()
    q_final = q**4  # 240 degree rotation about y

    q.slerp(q_final, t)  # Batch of interpolation
    assert q.slerp(q_final, t).shape == (6, 4)  # 4 quaternions of shape (4,)


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


def test_imag_real():
    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    real_part = q.real
    imag_part = q.imag
    assert torch.allclose(real_part, Quaternion(1.0, 0.0, 0.0, 0.0))
    assert torch.allclose(imag_part, Quaternion(0.0, 2.0, 3.0, 4.0))

    q_batch = Quaternion(
        torch.tensor([1.0, 0.0]),
        torch.tensor([2.0, 1.0]),
        torch.tensor([3.0, 2.0]),
        torch.tensor([4.0, 3.0]),
    )
    real_part_batch = q_batch.real
    imag_part_batch = q_batch.imag
    assert torch.allclose(
        real_part_batch, torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    )
    assert torch.allclose(
        imag_part_batch, torch.tensor([[0.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]])
    )


def test_from_rotation_matrix_under_edge_case_when_R_is_symmetric():
    matrix = torch.tensor(
        [
            [
                [-0.9925, -0.1025, 0.0669],
                [-0.1022, 0.3934, -0.9137],
                [0.0673, -0.9136, -0.4009],
            ],
            [
                [-1.0000, 0.0000, 0.0000],
                [0.0000, -1.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
            ],
            [
                [0.0000, 1.0000, 0.0000],
                [-1.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
            ],
            torch.eye(3),
        ]
    )
    q = Quaternion.from_rotation_matrix(matrix)

    q_0 = Quaternion(
        -0.00012604775963941,
        -0.0613069073653219,
        0.834683917470865,
        -0.5473063174646830,
    )
    q_1 = Quaternion(
        0.0,
        0.0,
        0.0,
        1.0,
    )
    q_2 = Quaternion(
        0.7071067690849304,
        0.0,
        0.0,
        -0.7071067690849304,
    )
    q_3 = Quaternion(
        1.0,
        0.0,
        0.0,
        0.0,
    )

    expected_quaternion = Quaternion(torch.stack([q_0, q_1, q_2, q_3]))
    assert torch.all(
        torch.all(
            torch.isclose(
                q.normalize() / expected_quaternion,
                Quaternion(1.0, 0.0, 0.0, 0.0),
                atol=1e-3,
            ),
            dim=1,
        )
        | torch.all(
            torch.isclose(
                q.normalize() / expected_quaternion,
                Quaternion(-1.0, 0.0, 0.0, 0.0),
                atol=1e-3,
            ),
            dim=1,
        )
    ).item()


def test_from_rotation_matrix_under_edge_case_when_R_is_symmetric_and_trace_is_negative():
    matrix = torch.tensor(
        [
            [
                # almost symmetric
                [-0.9925, -0.1025, 0.0669],
                [-0.1022, 0.3934, -0.9137],
                [0.0673, -0.9136, -0.4009],
            ],
            [
                # negative-definite (tr < 0)
                [-0.3333333, 0.6666667, -0.6666667],
                [0.6666667, -0.3333333, -0.6666667],
                [-0.6666667, -0.6666667, -0.3333333],
            ],
            [
                # not problematic
                [0.0000, 1.0000, 0.0000],
                [-1.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
            ],
            [
                # negative trace but not symmetric
                [-0.2440169, 0.9106836, -0.3333333],
                [0.3333333, -0.2440169, -0.9106836],
                [-0.9106836, -0.3333333, -0.2440169],
            ],
        ]
    )
    q = Quaternion.from_rotation_matrix(matrix)

    q_0 = Quaternion(
        -0.00012604775963941,
        -0.0613069073653219,
        0.834683917470865,
        -0.5473063174646830,
    )
    q_1 = Quaternion(
        0,
        -0.5773503,
        -0.5773503,
        0.5773503,
    )
    q_2 = Quaternion(
        0.7071067690849304,
        0.0,
        0.0,
        -0.7071067690849304,
    )
    q_3 = Quaternion(
        -0.258819,
        -0.5576775,
        -0.5576775,
        0.5576775,
    )

    # division between expected and computed should be close to 1 or -1, as these are the same quaternion.
    expected_quaternion = Quaternion(torch.stack([q_0, q_1, q_2, q_3]))
    assert torch.all(
        torch.all(
            torch.isclose(
                q.normalize() / expected_quaternion,
                Quaternion(1.0, 0.0, 0.0, 0.0),
                atol=1e-3,
            ),
            dim=1,
        )
        | torch.all(
            torch.isclose(
                q.normalize() / expected_quaternion,
                Quaternion(-1.0, 0.0, 0.0, 0.0),
                atol=1e-3,
            ),
            dim=1,
        )
    ).item()


def test_from_rotation_matrix_returns_correct_quaternion_for_a_spherical_combination_of_rotations():
    axis = torch.randn((100, 1, 3), generator=torch.Generator().manual_seed(0))
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    q_0 = Quaternion.from_axis_angle(axis, torch.tensor(99 * math.pi / 100))
    q_1 = Quaternion.from_axis_angle(axis, torch.tensor(101 * math.pi / 100))
    q_span = q_0.slerp(q_1, torch.linspace(0, 1, steps=100000)[..., None])

    R_span = q_span.to_rotation_matrix()
    q_from_R = Quaternion.from_rotation_matrix(R_span)
    error_norm = torch.minimum(
        (q_span - q_from_R).norm(dim=-1), (q_span + q_from_R).norm(dim=-1)
    )

    assert error_norm.max() < 5e-3, (
        f"Quaternion from rotation matrix does not match original quaternion span, problematic matrix at position({error_norm.argmax()}): {R_span.reshape(-1, 3, 3)[error_norm.argmax()]}"
    )
