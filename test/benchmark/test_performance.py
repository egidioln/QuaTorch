import pytest
import torch

from quatorch.quaternion import Quaternion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.benchmark(
    group=str(DEVICE),
    warmup=False,
)
def test_performance_slerp(benchmark):
    q1 = Quaternion(torch.randn(10000000, 4, device=DEVICE))
    q2 = Quaternion(torch.randn(10000000, 4, device=DEVICE))
    t = torch.rand(2, 1, 1, device=DEVICE)

    def slerp():
        x = Quaternion.slerp(q1, q2, t)[0, 0, 0].item()

    result = benchmark(
        slerp,
    )


@pytest.mark.benchmark(
    group=str(DEVICE),
    warmup=False,
)
def test_performance_rotate_vector(benchmark):
    q1 = Quaternion(torch.randn(10000000, 4, device=DEVICE))
    vectors = torch.randn(10000000, 3, device=DEVICE)

    def rotate_vector():
        x = Quaternion.rotate_vector(q1, vectors)[0, 0].item()

    result = benchmark(
        rotate_vector,
    )


@pytest.mark.benchmark(
    group=str(DEVICE),
    warmup=False,
)
def test_performance_multiplication(benchmark):
    q1 = Quaternion(torch.randn(10000000, 4, device=DEVICE))
    q2 = Quaternion(torch.randn(10000000, 4, device=DEVICE))

    def multiplication():
        x = Quaternion.mul(q1, q2)[0, 0].item()

    result = benchmark(
        multiplication,
    )
