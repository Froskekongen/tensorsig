# Reproducing benchmarks

Write new measurements under ignored `benchmarks/results/` or outside the
checkout. Keep result files, logs, profiler traces, and source archives out of
commits. Small deterministic benchmark inputs live in `benchmarks/fixtures/`;
the exact graph reference contains rational values and a survival flag, with
no timing, memory, or machine-specific data.

Install the development dependencies with `uv sync --locked`, then, for example:

```console
uv run python benchmarks/benchmark_pascal_algorithms.py --quick --dtype float64 --output benchmarks/results/pascal.json
uv run python benchmarks/benchmark_rolling_logsignature.py --help
uv run python benchmarks/benchmark_exact_compilers.py --help
uv run python scripts/confirm_exact_graph_memory.py --output benchmarks/results/exact-graph-memory.json
```

Inspect each program's options before launching a large run. Record the source
revision, parameters, hardware, dtype, thread settings, warmups, synchronization,
and repeated samples alongside results. Forward agreement, gradient agreement,
latency, and memory are separate measurements. A CPU run does not validate an
accelerator or establish a universal performance crossover.
