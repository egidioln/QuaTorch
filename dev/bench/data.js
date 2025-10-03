window.BENCHMARK_DATA = {
  "lastUpdate": 1759520438957,
  "repoUrl": "https://github.com/egidioln/QuaTorch",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "lucasegidio1@gmail.com",
            "name": "Lucas Egidio",
            "username": "egidioln"
          },
          "committer": {
            "email": "lucasegidio1@gmail.com",
            "name": "Lucas Egidio",
            "username": "egidioln"
          },
          "distinct": true,
          "id": "b97ea48f138c9503f2e055058be5aa362635461e",
          "message": "ci: give write permissions to docs job",
          "timestamp": "2025-10-03T21:39:04+02:00",
          "tree_id": "60421c12137e50f4187dc78008e6be28258b5b54",
          "url": "https://github.com/egidioln/QuaTorch/commit/b97ea48f138c9503f2e055058be5aa362635461e"
        },
        "date": 1759520437844,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/benchmark/test_performance.py::test_performance_slerp",
            "value": 0.6560955575906878,
            "unit": "iter/sec",
            "range": "stddev: 0.008958403338576263",
            "extra": "mean: 1.52416822280004 sec\nrounds: 5"
          },
          {
            "name": "test/benchmark/test_performance.py::test_performance_rotate_vector",
            "value": 1.5527506609647037,
            "unit": "iter/sec",
            "range": "stddev: 0.009077292293020772",
            "extra": "mean: 644.0184023999791 msec\nrounds: 5"
          },
          {
            "name": "test/benchmark/test_performance.py::test_performance_multiplication",
            "value": 3.9078075543961814,
            "unit": "iter/sec",
            "range": "stddev: 0.003850788313605824",
            "extra": "mean: 255.89796479998768 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}