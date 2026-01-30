import pytest
import torch

from quatorch.quaternion import Quaternion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    if DEVICE.type == "cpu":
        torch.cpu.synchronize()


@pytest.mark.benchmark(
    group=str(DEVICE),
    warmup=False,
)
def test_performance_slerp(benchmark):
    q1 = Quaternion(torch.randn(10000000, 4, device=DEVICE))
    q2 = Quaternion(torch.randn(10000000, 4, device=DEVICE))
    t = torch.rand(2, 1, 1, device=DEVICE)

    def slerp():
        x = Quaternion.slerp(q1, q2, t)[0, 0, 0]
        synchronize()

    result = benchmark(
        slerp,
    )


@pytest.mark.parametrize("num_points", [1000000, 5000000, 10000000])
@pytest.mark.benchmark(
    group=str(DEVICE),
    warmup=False,
)
def test_performance_rotate_vector(benchmark, num_points):
    q1 = Quaternion(torch.randn(num_points, 4, device=DEVICE))
    vectors = torch.randn(num_points, 3, device=DEVICE)

    def rotate_vector():
        x = Quaternion.rotate_vector(q1, vectors)
        synchronize()

    result = benchmark(
        rotate_vector,
    )


def multiplication(q1: Quaternion, q2: Quaternion) -> Quaternion:
    return q1 * q2


multiplication_compiled = torch.compile(
    multiplication,
    fullgraph=True,
    # mode="max-autotune",  # The execution time increases by 3x with this option
)


@pytest.mark.parametrize(
    "multiplication",
    [multiplication, multiplication_compiled],
)
@pytest.mark.benchmark(
    group=str(DEVICE),
    warmup=False,
)
def test_performance_multiplication(benchmark, multiplication):
    q1 = Quaternion(torch.randn(10000000, 4, device=DEVICE))
    q2 = Quaternion(torch.randn(10000000, 4, device=DEVICE))

    for warmup_n in range(2):
        q = multiplication(q1, q2)
        synchronize()

    synchronize()
    result = benchmark(
        multiplication,
        q1,
        q2,
    )


def test_compile_multiplication_match():
    q1 = Quaternion(torch.randn(10000, 4, device=DEVICE))
    q2 = Quaternion(torch.randn(10000, 4, device=DEVICE))
    result_compiled = multiplication_compiled(q1, q2)
    result = multiplication(q1, q2)

    assert torch.allclose(result, result_compiled, atol=1e-6)
