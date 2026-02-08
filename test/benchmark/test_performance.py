import pytest
import quaternion as np_quat
import quaternionic
import torch

from quatorch.quaternion import Quaternion

# We use an annotation "convert_input" to specify how to convert the input for the timed function should be converted before the timed functions execution.


def annotate_convert_input(func, convert_input=lambda x: x):
    func.__annotations__["convert_input"] = convert_input
    return func


def _synchronize(data: torch.Tensor):
    device = data.device
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
        return
    if torch.device(device).type == "cpu":
        torch.cpu.synchronize()
        return
    raise RuntimeError("Unsupported device")


def _to_numpy(data: torch.Tensor):
    data_numpy = data.cpu().numpy()
    if data_numpy.ndim == 2 and data_numpy.shape[1] == 4:
        return np_quat.as_quat_array(data_numpy)
    return data_numpy


def _to_quaternionic(data: torch.Tensor):
    data_numpy = data.cpu().numpy()
    if data_numpy.shape[-1] == 4:
        return quaternionic.array(data_numpy)
    return data_numpy


slerp_compiled = torch.compile(Quaternion.slerp, fullgraph=True)


@pytest.mark.parametrize(
    "slerp",
    [
        pytest.param(
            annotate_convert_input(Quaternion.slerp, lambda x: x.cpu()),
            id="cpu_eager",
        ),
        pytest.param(
            annotate_convert_input(
                lambda *_: slerp_compiled(*_),
                lambda x: x.cpu(),
            ),
            id="cpu_compiled",
        ),
        pytest.param(
            annotate_convert_input(
                lambda *_: slerp_compiled(*_),
                lambda x: x.cuda(),
            ),
            id="cuda_compiled",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
        pytest.param(
            annotate_convert_input(
                Quaternion.slerp,
                lambda x: x.cuda(),
            ),
            id="cuda_eager",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
        # quaternionic
        pytest.param(
            annotate_convert_input(
                lambda q1, q2, t: quaternionic.slerp(q1, q2, t),
                _to_quaternionic,
            ),
            id="quaternionic",
        ),
    ],
)
@pytest.mark.benchmark(
    group="test_performance_slerp",
    warmup=True,
)
def test_performance_slerp(benchmark, slerp):
    q1 = Quaternion(torch.randn(10000000, 4))
    q2 = Quaternion(torch.randn(10000000, 4))
    t = torch.rand(2, 1, 1)

    def slerp_fn():
        x = slerp(q1, q2, t)
        _synchronize(x)

    result = benchmark(slerp_fn)


# Benchmark Rotate Vector


compiled_rotate_vector = torch.compile(
    Quaternion.rotate_vector,
    fullgraph=True,
)


rotate_numpy = annotate_convert_input(np_quat.rotate_vectors, convert_input=_to_numpy)


@pytest.mark.parametrize(
    "rotate",
    [
        pytest.param(
            annotate_convert_input(Quaternion.rotate_vector, lambda x: x.cpu()),
            id="cpu_eager",
        ),
        pytest.param(
            annotate_convert_input(
                lambda *_: compiled_rotate_vector(*_),
                lambda x: x.cpu(),
            ),
            id="cpu_compiled",
        ),
        pytest.param(
            annotate_convert_input(
                lambda *_: compiled_rotate_vector(*_),
                lambda x: x.cuda(),
            ),
            id="cuda_compiled",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
        pytest.param(
            annotate_convert_input(
                lambda *_: Quaternion.rotate_vector(*_),
                lambda x: x.cuda(),
            ),
            id="cuda_eager",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
        pytest.param(
            rotate_numpy,
            id="numpy",
        ),
    ],
)
@pytest.mark.benchmark(
    group="test_performance_rotate_vector",
    warmup=True,
)
def test_performance_rotate_vector(benchmark, rotate, num_points=10000000):
    q1 = Quaternion(torch.randn(1, 4))
    vectors = torch.randn(num_points, 3)

    convert_input = rotate.__annotations__.get("convert_input", lambda x: x)
    q1 = convert_input(q1)
    vectors = convert_input(vectors)

    def rotate_vector():
        x = rotate(q1, vectors)
        _synchronize(x)

    result = benchmark(
        rotate_vector,
    )


def multiplication(q1: Quaternion, q2: Quaternion) -> Quaternion:
    return Quaternion.mul(q1, q2)


multiplication_compiled = torch.compile(
    multiplication,
    fullgraph=True,
)


def multiplication_numpy(
    q1: np_quat.quaternion, q2: np_quat.quaternion
) -> np_quat.quaternion:
    result_np = q1 * q2
    return result_np


@pytest.mark.parametrize(
    "multiplication",
    [
        pytest.param(
            annotate_convert_input(lambda *_: multiplication(*_), lambda x: x.cpu()),
            id="cpu_eager",
        ),
        pytest.param(
            annotate_convert_input(lambda *_: multiplication(*_), lambda x: x.cuda()),
            id="cuda_eager",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
        pytest.param(
            annotate_convert_input(
                lambda *_: multiplication_compiled(*_),
                lambda x: x.cpu(),
            ),
            id="cpu_compiled",
        ),
        pytest.param(
            annotate_convert_input(
                lambda *_: multiplication_compiled(*_),
                lambda x: x.cuda(),
            ),
            id="cuda_compiled",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
        pytest.param(
            annotate_convert_input(lambda q1, q2: q1 * q2, convert_input=_to_numpy),
            id="numpy",
        ),
    ],
)
@pytest.mark.benchmark(
    group="test_performance_multiplication",
    warmup=True,
)
def test_performance_multiplication(benchmark, multiplication):
    q1 = Quaternion(torch.randn(10_000_000, 4))
    q2 = Quaternion(torch.randn(10_000_000, 4))

    convert_input = multiplication.__annotations__.get("convert_input", lambda x: x)
    q1 = convert_input(q1)
    q2 = convert_input(q2)

    def multiplication_fn():
        x = multiplication(q1, q2)
        _synchronize(x)

    for warmup_n in range(2):
        multiplication_fn()

    result = benchmark(
        multiplication_fn,
    )


def test_compile_multiplication_match():
    q1 = Quaternion(torch.randn(10000, 4))
    q2 = Quaternion(torch.randn(10000, 4))
    result_compiled = multiplication_compiled(q1, q2)
    result = multiplication(q1, q2)

    assert torch.allclose(result, result_compiled, atol=1e-6)
