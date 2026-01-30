import pytest
import torch

from quatorch.quaternion import Quaternion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    if DEVICE.type == "cpu":
        torch.cpu.synchronize()


@pytest.mark.parametrize(
    "slerp",
    [
        pytest.param(Quaternion.slerp, id="original"),
        pytest.param(
            torch.compile(
                Quaternion.slerp,
                fullgraph=True,
            ),
            id="compiled",
        ),
        pytest.param(
            torch.compile(
                Quaternion.slerp,
                fullgraph=True,
                mode="max-autotune",
            ),
            id="compiled_max_autotune",
        ),
    ],
)
@pytest.mark.benchmark(
    group=str(DEVICE),
    warmup=False,
)
def test_performance_slerp(benchmark, slerp):
    q1 = Quaternion(torch.randn(10000000, 4, device=DEVICE))
    q2 = Quaternion(torch.randn(10000000, 4, device=DEVICE))
    t = torch.rand(2, 1, 1, device=DEVICE)

    def slerp_fn():
        x = slerp(q1, q2, t)
        synchronize()

    result = benchmark(
        slerp_fn,
    )


@pytest.mark.parametrize(
    "rotate",
    [
        pytest.param(Quaternion.rotate_vector, id="original"),
        pytest.param(
            torch.compile(
                Quaternion.rotate_vector,
                fullgraph=True,
            ),
            id="compiled",
        ),
    ],
)
@pytest.mark.benchmark(
    group=str(DEVICE),
    warmup=False,
)
def test_performance_rotate_vector(benchmark, rotate, num_points=10000000):
    q1 = Quaternion(torch.randn(num_points, 4, device=DEVICE))
    vectors = torch.randn(num_points, 3, device=DEVICE)

    def rotate_vector():
        x = rotate(q1, vectors)
        synchronize()

    result = benchmark(
        rotate_vector,
    )


def multiplication(q1: Quaternion, q2: Quaternion) -> Quaternion:
    return Quaternion.mul(q1, q2)


multiplication_compiled = torch.compile(
    multiplication,
    fullgraph=True,
)

multiplication_compiled_max_autotune = torch.compile(
    multiplication,
    fullgraph=True,
    mode="max-autotune",
)


@pytest.mark.parametrize(
    "multiplication",
    [
        pytest.param(multiplication, id="original"),
        pytest.param(multiplication_compiled, id="compiled"),
        pytest.param(multiplication_compiled_max_autotune, id="compiled_max_autotune"),
    ],
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
