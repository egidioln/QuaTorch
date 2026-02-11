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


@pytest.mark.parametrize("num_quaternions", [1_000, 10_000, 100_000])
@pytest.mark.parametrize(
    "slerp",
    [
        pytest.param(
            annotate_convert_input(lambda *_: Quaternion.slerp(*_), lambda x: x.cpu()),
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
                lambda *_: Quaternion.slerp(*_),
                lambda x: x.cuda(),
            ),
            id="cuda_eager",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
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
def test_performance_slerp(benchmark, slerp, num_quaternions):
    q1 = Quaternion(torch.randn(num_quaternions, 4))
    q2 = Quaternion(torch.randn(num_quaternions, 4))
    t = torch.rand(100, 1, 1)

    def slerp_fn():
        x = slerp(q1, q2, t)
        _synchronize(x)

    _ = benchmark(slerp_fn)


# Benchmark Rotate Vector


compiled_rotate_vector = torch.compile(
    Quaternion.rotate_vector,
    fullgraph=True,
)


rotate_numpy = annotate_convert_input(np_quat.rotate_vectors, convert_input=_to_numpy)


@pytest.mark.parametrize("num_quaternions", [1_000, 100_000, 10_000_000])
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
        pytest.param(
            annotate_convert_input(
                lambda q, v: q.rotate(v), convert_input=_to_quaternionic
            ),
            id="quaternionic",
        ),
    ],
)
@pytest.mark.benchmark(
    group="test_performance_rotate_vector",
    warmup=True,
)
def test_performance_rotate_vector(benchmark, rotate, num_quaternions):
    q1 = Quaternion(torch.randn(num_quaternions, 4))
    vectors = torch.randn(1, 3)

    convert_input = rotate.__annotations__.get("convert_input", lambda x: x)
    q1 = convert_input(q1)
    vectors = convert_input(vectors)

    def rotate_vector():
        x = rotate(q1, vectors)
        _synchronize(x)

    _ = benchmark(
        rotate_vector,
    )


multiplication_compiled = torch.compile(
    Quaternion.mul,
    fullgraph=True,
)


def multiplication_numpy(
    q1: np_quat.quaternion, q2: np_quat.quaternion
) -> np_quat.quaternion:
    result_np = q1 * q2
    return result_np


@pytest.mark.parametrize("num_quaternions", [1_000, 100_000, 10_000_000])
@pytest.mark.parametrize(
    "multiplication",
    [
        pytest.param(
            annotate_convert_input(lambda *_: Quaternion.mul(*_), lambda x: x.cpu()),
            id="cpu_eager",
        ),
        pytest.param(
            annotate_convert_input(lambda *_: Quaternion.mul(*_), lambda x: x.cuda()),
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
        pytest.param(
            annotate_convert_input(
                lambda q1, q2: q1 * q2, convert_input=_to_quaternionic
            ),
            id="quaternionic",
        ),
    ],
)
@pytest.mark.benchmark(
    group="test_performance_multiplication",
    warmup=True,
)
def test_performance_multiplication(
    benchmark,
    multiplication,
    num_quaternions,
):
    q1 = Quaternion(torch.randn(num_quaternions, 4))
    q2 = Quaternion(torch.randn(num_quaternions, 4))

    convert_input = multiplication.__annotations__.get("convert_input", lambda x: x)
    q1 = convert_input(q1)
    q2 = convert_input(q2)

    def multiplication_fn():
        x = multiplication(q1, q2)
        _synchronize(x)

    for warmup_n in range(2):
        multiplication_fn()

    _ = benchmark(
        multiplication_fn,
    )


def test_compile_multiplication_match():
    q1 = Quaternion(torch.randn(10000, 4))
    q2 = Quaternion(torch.randn(10000, 4))
    result_compiled = multiplication_compiled(q1, q2)
    result = q1 * q2

    assert torch.allclose(result, result_compiled, atol=1e-6)


def plot_benchmark_results(benchmark_file):
    """
    Read benchmark results from benchmark_output.json and create plots.
    This test should be run after all benchmarks have completed.
    """
    import json

    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib is required for plotting benchmark results")

    if not benchmark_file.exists():
        pytest.skip(f"Benchmark output file not found: {benchmark_file}")

    # Load benchmark data
    benchmark_data = json.loads(benchmark_file.read_text())

    # Group benchmarks by test group
    benchmark_groups = {}
    for benchmark in benchmark_data.get("benchmarks", []):
        group = benchmark.get("group", "default")
        if group not in benchmark_groups:
            benchmark_groups[group] = []
        benchmark_groups[group].append(benchmark)

    # Create plots for each benchmark group
    for group_name, benchmarks in benchmark_groups.items():
        _plot_benchmark_group(group_name, benchmarks, benchmark_file.parent)


def _plot_benchmark_group(group_name, benchmarks, output_dir):
    """
    Create a line plot for a specific benchmark group.

    Args:
        group_name: Name of the benchmark group
        benchmarks: List of benchmark results
        output_dir: Directory to save the plot
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    # Parse benchmark names and organize data by method
    data = {}

    for benchmark in benchmarks:
        name = benchmark["name"]
        stats = benchmark["stats"]

        # Parse the benchmark name
        # Format: test_name[method-size] or test_name[method]
        if "[" in name and "]" in name:
            params = name[name.index("[") + 1 : name.index("]")]
            parts = params.split("-")

            if len(parts) == 2:
                method, size_str = parts
                try:
                    size = int(size_str)
                except ValueError:
                    # Not a parameterized size test, skip
                    continue
            elif len(parts) == 1:
                # No size parameter
                method = parts[0]
                size = None
            else:
                continue

            if method not in data:
                data[method] = {"x": [], "min": [], "mean": [], "max": []}

            if size is not None:
                # Convert times from seconds to microseconds
                min_time = stats["min"] * 1e6
                mean_time = stats["mean"] * 1e6
                max_time = stats["max"] * 1e6

                data[method]["x"].append(size)
                data[method]["min"].append(min_time)
                data[method]["mean"].append(mean_time)
                data[method]["max"].append(max_time)

    # Skip if no parameterized data found
    if not data or all(len(d["x"]) == 0 for d in data.values()):
        return

    # Sort each method's data by x values
    for method in data:
        if data[method]["x"]:
            indices = np.argsort(data[method]["x"])
            data[method]["x"] = [data[method]["x"][i] for i in indices]
            data[method]["min"] = [data[method]["min"][i] for i in indices]
            data[method]["mean"] = [data[method]["mean"][i] for i in indices]
            data[method]["max"] = [data[method]["max"][i] for i in indices]

    # Collect all unique sizes across all methods
    all_sizes = sorted(set(size for method in data.values() for size in method["x"]))

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color scheme for each method
    colors = {
        "numpy": "#1f77b4",
        "cpu_eager": "#ff7f0e",
        "cpu_compiled": "#2ca02c",
        "cuda_eager": "#d62728",
        "cuda_compiled": "#9467bd",
        "quaternionic": "#8c564b",
    }
    display_names = {
        "numpy": "numpy-quaternion",
        "cpu_eager": "quatorch (CPU Eager)",
        "cpu_compiled": "quatorch (CPU Compiled)",
        "cuda_eager": "quatorch (CUDA Eager)",
        "cuda_compiled": "quatorch (CUDA Compiled)",
        "quaternionic": "quaternionic",
    }

    # Default colors for methods not in the predefined list
    default_colors = plt.cm.tab10.colors
    color_idx = 0

    methods = sorted(data.keys())
    num_methods = len(methods)

    # Calculate bar width and positions
    bar_width = 0.8 / num_methods
    x_positions = np.arange(len(all_sizes))

    # Plot bars for each method
    for idx, method in enumerate(methods):
        if data[method]["x"]:
            # Get color for this method
            if method in colors:
                color = colors[method]
            else:
                color = default_colors[color_idx % len(default_colors)]
                color_idx += 1

            # Create arrays for this method aligned with all_sizes
            means = []
            errors_lower = []
            errors_upper = []

            for size in all_sizes:
                if size in data[method]["x"]:
                    size_idx = data[method]["x"].index(size)
                    mean_val = data[method]["mean"][size_idx]
                    min_val = data[method]["min"][size_idx]
                    max_val = data[method]["max"][size_idx]

                    means.append(mean_val)
                    errors_lower.append(mean_val - min_val)
                    errors_upper.append(max_val - mean_val)
                else:
                    means.append(np.nan)
                    errors_lower.append(0)
                    errors_upper.append(0)

            means = np.array(means)
            errors = np.array([errors_lower, errors_upper])

            # Calculate x positions for this method's bars
            x_pos = x_positions + (idx - num_methods / 2 + 0.5) * bar_width

            # Plot bars with error bars (whiskers)
            ax.bar(
                x_pos,
                means,
                bar_width,
                label=display_names.get(method, method),
                color=color,
                alpha=0.8,
                yerr=errors,
                capsize=3,
                error_kw={"linewidth": 1, "elinewidth": 1},
            )

    ax.set_yscale("log")
    ax.set_xlabel("Size (number of elements)", fontsize=12)
    ax.set_ylabel("Execution Time (μs)", fontsize=12)

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{size:,}" for size in all_sizes], rotation=45, ha="right")

    # Format the title
    title = group_name.replace("test_performance_", "").replace("_", " ").title()
    ax.set_title(f"{title} Performance Benchmark", fontsize=14, fontweight="bold")

    ax.set_title(f"{title} Performance Benchmark", fontsize=14, fontweight="bold")

    ax.legend(loc="best", fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.3, axis="y")
    plt.tight_layout()

    # Save the plot
    output_file = Path(output_dir) / f"{group_name}_benchmark.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved as '{output_file}'")


def compute_performance_comparison(benchmark_file):
    """
    Compute geometric mean of performance improvements between cpu_compiled and other methods.

    Args:
        benchmark_file: Path to the benchmark output JSON file
    """
    import json

    import numpy as np

    if not benchmark_file.exists():
        print(f"Benchmark output file not found: {benchmark_file}")
        return

    # Load benchmark data
    benchmark_data = json.loads(benchmark_file.read_text())

    # Organize data by method and size
    results = {}  # {group: {method: {size: mean_time}}}

    for benchmark in benchmark_data.get("benchmarks", []):
        name = benchmark["name"]
        group = benchmark.get("group", "default")
        stats = benchmark["stats"]

        # Parse the benchmark name
        if "[" in name and "]" in name:
            params = name[name.index("[") + 1 : name.index("]")]
            parts = params.split("-")

            if len(parts) == 2:
                method, size_str = parts
                try:
                    size = int(size_str)
                except ValueError:
                    continue

                if group not in results:
                    results[group] = {}
                if method not in results[group]:
                    results[group][method] = {}

                # Store mean time in microseconds
                results[group][method][size] = stats["mean"] * 1e6

    # Compute geometric mean of speedup ratios
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: cpu_compiled vs other methods")
    print("=" * 80)

    for group, methods_data in results.items():
        if "cpu_compiled" not in methods_data:
            continue

        print(f"\n{group.replace('test_performance_', '').replace('_', ' ').title()}:")
        print("-" * 80)

        cpu_compiled = methods_data["cpu_compiled"]

        # Compare against numpy and quaternionic
        for compare_method in ["numpy", "quaternionic"]:
            if compare_method not in methods_data:
                continue

            compare_data = methods_data[compare_method]

            # Find common sizes
            common_sizes = sorted(set(cpu_compiled.keys()) & set(compare_data.keys()))

            if not common_sizes:
                continue

            ratios = []
            print(f"\n  cpu_compiled vs {compare_method}:")

            for size in common_sizes:
                cpu_time = cpu_compiled[size]
                compare_time = compare_data[size]
                speedup = compare_time / cpu_time
                ratios.append(speedup)

                if speedup > 1:
                    print(f"    Size {size:>10,}: {speedup:6.2f}x faster")
                else:
                    print(f"    Size {size:>10,}: {1 / speedup:6.2f}x slower")

            # Compute geometric mean
            if ratios:
                geom_mean = np.exp(np.mean(np.log(ratios)))
                print(f"\n  Geometric mean speedup: {geom_mean:.2f}x")

                if geom_mean > 1:
                    print(
                        f"  → cpu_compiled is {geom_mean:.2f}x faster than {compare_method} on average"
                    )
                else:
                    print(
                        f"  → cpu_compiled is {1 / geom_mean:.2f}x slower than {compare_method} on average"
                    )

    print("\n" + "=" * 80)
    print("Note: Geometric mean is computed across all common sizes.")
    print("Speedup > 1 means cpu_compiled is faster, < 1 means it's slower.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "benchmark_file",
        type=Path,
        help="Path to the benchmark output JSON file",
    )
    args = parser.parse_args()

    plot_benchmark_results(args.benchmark_file)
    compute_performance_comparison(args.benchmark_file)
