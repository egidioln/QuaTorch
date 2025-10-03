window.BENCHMARK_DATA = {
  "lastUpdate": 1759518008403,
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
          "id": "5a66e90e321848edd861936b79a5e5b2d8d6812d",
          "message": "ci: add a ruff format job",
          "timestamp": "2025-10-03T20:58:48+02:00",
          "tree_id": "e23be47b4c8b527c95b27ea0dafe8739c32d76e6",
          "url": "https://github.com/egidioln/QuaTorch/commit/5a66e90e321848edd861936b79a5e5b2d8d6812d"
        },
        "date": 1759518007714,
        "tool": "pytest",
        "benches": [
          {
            "name": "test/benchmark/test_performance.py::test_performance_slerp",
            "value": 0.6817583035441822,
            "unit": "iter/sec",
            "range": "stddev: 0.0024801435583390523",
            "extra": "mean: 1.4667954828000034 sec\nrounds: 5"
          },
          {
            "name": "test/benchmark/test_performance.py::test_performance_rotate_vector",
            "value": 1.6284899597133764,
            "unit": "iter/sec",
            "range": "stddev: 0.0017181220904760853",
            "extra": "mean: 614.0658062 msec\nrounds: 5"
          },
          {
            "name": "test/benchmark/test_performance.py::test_performance_multiplication",
            "value": 4.045735008163204,
            "unit": "iter/sec",
            "range": "stddev: 0.0013814939022863983",
            "extra": "mean: 247.17387519999932 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}