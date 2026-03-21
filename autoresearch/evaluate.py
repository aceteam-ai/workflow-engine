#!/usr/bin/env python3
"""
Autoresearch benchmark harness (READ-ONLY — do not modify).

Builds and executes 4 benchmark workflows programmatically, measuring:
  - Wall-clock execution time (seconds)
  - Peak memory usage (MB)
  - Correctness (output matches expected values)

All workflows use pure arithmetic nodes — no LLM calls, no API keys,
fully deterministic and reproducible.

Usage:
    uv run python autoresearch/evaluate.py
"""

import asyncio
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engine import (
    AddNode,
    ConstantIntegerNode,
    Edge,
    FloatValue,
    InMemoryContext,
    InputNode,
    IntegerValue,
    OutputNode,
    ParallelExecutionAlgorithm,
    SequenceValue,
    SumNode,
    TopologicalExecutionAlgorithm,
    Workflow,
    WorkflowExecutionResultStatus,
)


# ---------------------------------------------------------------------------
# Benchmark workflow builders
# ---------------------------------------------------------------------------


def build_linear_5() -> tuple[Workflow, dict, dict]:
    """A → B → C → D → E: 5 chained additions.

    Input: x=1
    Chain: const(10) + x → +const(20) → +const(30) → +const(40) → output
    Expected: 1 + 10 + 20 + 30 + 40 = 101
    """
    input_node = InputNode.from_fields(x=IntegerValue)
    output_node = OutputNode.from_fields(result=IntegerValue)

    c1 = ConstantIntegerNode.from_value(id="c1", value=10)
    c2 = ConstantIntegerNode.from_value(id="c2", value=20)
    c3 = ConstantIntegerNode.from_value(id="c3", value=30)
    c4 = ConstantIntegerNode.from_value(id="c4", value=40)

    add1 = AddNode(id="add1")
    add2 = AddNode(id="add2")
    add3 = AddNode(id="add3")
    add4 = AddNode(id="add4")

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[c1, c2, c3, c4, add1, add2, add3, add4],
        edges=[
            # add1 = x + c1
            Edge.from_nodes(source=input_node, source_key="x", target=add1, target_key="a"),
            Edge.from_nodes(source=c1, source_key="value", target=add1, target_key="b"),
            # add2 = add1 + c2
            Edge.from_nodes(source=add1, source_key="sum", target=add2, target_key="a"),
            Edge.from_nodes(source=c2, source_key="value", target=add2, target_key="b"),
            # add3 = add2 + c3
            Edge.from_nodes(source=add2, source_key="sum", target=add3, target_key="a"),
            Edge.from_nodes(source=c3, source_key="value", target=add3, target_key="b"),
            # add4 = add3 + c4
            Edge.from_nodes(source=add3, source_key="sum", target=add4, target_key="a"),
            Edge.from_nodes(source=c4, source_key="value", target=add4, target_key="b"),
            # output
            Edge.from_nodes(source=add4, source_key="sum", target=output_node, target_key="result"),
        ],
    )

    input_data = {"x": IntegerValue(1)}
    expected = {"result": 1 + 10 + 20 + 30 + 40}
    return workflow, input_data, expected


def build_branching_10() -> tuple[Workflow, dict, dict]:
    """Input → 3 parallel branches → merge → output (10 nodes total).

    Input: x=5
    Branch A: x + const(100) = 105
    Branch B: x + const(200) = 205
    Branch C: x + const(300) = 305
    Merge: 105 + 205 + 305 = 615

    Nodes: input, 3 constants, 3 branch-adds, 1 merge-add(arity=3), output = 10 total
    """
    input_node = InputNode.from_fields(x=IntegerValue)
    output_node = OutputNode.from_fields(result=IntegerValue)

    ca = ConstantIntegerNode.from_value(id="ca", value=100)
    cb = ConstantIntegerNode.from_value(id="cb", value=200)
    cc = ConstantIntegerNode.from_value(id="cc", value=300)

    branch_a = AddNode(id="branch_a")
    branch_b = AddNode(id="branch_b")
    branch_c = AddNode(id="branch_c")

    merge = AddNode.with_arity(id="merge", arity=3)

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[ca, cb, cc, branch_a, branch_b, branch_c, merge],
        edges=[
            # Branch A: x + 100
            Edge.from_nodes(source=input_node, source_key="x", target=branch_a, target_key="a"),
            Edge.from_nodes(source=ca, source_key="value", target=branch_a, target_key="b"),
            # Branch B: x + 200
            Edge.from_nodes(source=input_node, source_key="x", target=branch_b, target_key="a"),
            Edge.from_nodes(source=cb, source_key="value", target=branch_b, target_key="b"),
            # Branch C: x + 300
            Edge.from_nodes(source=input_node, source_key="x", target=branch_c, target_key="a"),
            Edge.from_nodes(source=cc, source_key="value", target=branch_c, target_key="b"),
            # Merge: branch_a + branch_b + branch_c
            Edge.from_nodes(source=branch_a, source_key="sum", target=merge, target_key="a"),
            Edge.from_nodes(source=branch_b, source_key="sum", target=merge, target_key="b"),
            Edge.from_nodes(source=branch_c, source_key="sum", target=merge, target_key="c"),
            # Output
            Edge.from_nodes(source=merge, source_key="sum", target=output_node, target_key="result"),
        ],
    )

    input_data = {"x": IntegerValue(5)}
    expected = {"result": (5 + 100) + (5 + 200) + (5 + 300)}
    return workflow, input_data, expected


def build_nested_20() -> tuple[Workflow, dict, dict]:
    """20-node workflow simulating nested sub-workflow depth.

    4 layers of 5 constants each, summed within each layer, then
    the 4 layer sums are added together.

    Layer 0: const(1..5) → add5_0 = 15
    Layer 1: const(6..10) → add5_1 = 40
    Layer 2: const(11..15) → add5_2 = 65
    Layer 3: const(16..20) → add5_3 = 90
    Final: 15 + 40 + 65 + 90 = 210

    Nodes: 20 constants + 4 layer-adds + 1 final-add + input + output = 27 total
    """
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(result=IntegerValue)

    all_constants = []
    layer_adds = []
    all_edges = []

    for layer in range(4):
        layer_constants = []
        for i in range(5):
            val = layer * 5 + i + 1
            node = ConstantIntegerNode.from_value(id=f"c_{layer}_{i}", value=val)
            layer_constants.append(node)
        all_constants.extend(layer_constants)

        layer_add = AddNode.with_arity(id=f"layer_add_{layer}", arity=5)
        layer_adds.append(layer_add)

        field_names = ["a", "b", "c", "d", "e"]
        for j, const in enumerate(layer_constants):
            all_edges.append(
                Edge.from_nodes(
                    source=const,
                    source_key="value",
                    target=layer_add,
                    target_key=field_names[j],
                )
            )

    final_add = AddNode.with_arity(id="final_add", arity=4)
    field_names_4 = ["a", "b", "c", "d"]
    for i, la in enumerate(layer_adds):
        all_edges.append(
            Edge.from_nodes(
                source=la,
                source_key="sum",
                target=final_add,
                target_key=field_names_4[i],
            )
        )
    all_edges.append(
        Edge.from_nodes(
            source=final_add,
            source_key="sum",
            target=output_node,
            target_key="result",
        )
    )

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[*all_constants, *layer_adds, final_add],
        edges=all_edges,
    )

    input_data = {}
    expected = {"result": sum(range(1, 21))}  # 210
    return workflow, input_data, expected


def build_parallel_fan() -> tuple[Workflow, dict, dict]:
    """Fan-out to 8 nodes, fan-in to aggregator.

    Input: x=7
    Fan-out: 8 branches each add a different constant (10, 20, ..., 80) to x
    Fan-in: Sum all 8 branch results via AddNode(arity=8)

    Expected: sum(x + k for k in [10,20,...,80]) = 8*7 + (10+20+...+80) = 56 + 360 = 416

    Nodes: 8 constants + 8 branch-adds + 1 fan-in-add + input + output = 19 total
    """
    input_node = InputNode.from_fields(x=IntegerValue)
    output_node = OutputNode.from_fields(result=IntegerValue)

    n_branches = 8
    constants = []
    branches = []
    edges = []

    for i in range(n_branches):
        c = ConstantIntegerNode.from_value(id=f"fan_c{i}", value=(i + 1) * 10)
        constants.append(c)

        b = AddNode(id=f"fan_b{i}")
        branches.append(b)

        edges.append(
            Edge.from_nodes(source=input_node, source_key="x", target=b, target_key="a")
        )
        edges.append(
            Edge.from_nodes(source=c, source_key="value", target=b, target_key="b")
        )

    fan_in = AddNode.with_arity(id="fan_in", arity=n_branches)

    # Connect branches to fan-in
    field_names_8 = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for i, b in enumerate(branches):
        edges.append(
            Edge.from_nodes(
                source=b,
                source_key="sum",
                target=fan_in,
                target_key=field_names_8[i],
            )
        )

    edges.append(
        Edge.from_nodes(
            source=fan_in,
            source_key="sum",
            target=output_node,
            target_key="result",
        )
    )

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[*constants, *branches, fan_in],
        edges=edges,
    )

    input_data = {"x": IntegerValue(7)}
    expected = {"result": sum(7 + (i + 1) * 10 for i in range(n_branches))}  # 416
    return workflow, input_data, expected


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

BENCHMARKS = [
    ("linear_5", build_linear_5),
    ("branching_10", build_branching_10),
    ("nested_20", build_nested_20),
    ("parallel_fan", build_parallel_fan),
]

ALGORITHMS = [
    ("topological", TopologicalExecutionAlgorithm),
    ("parallel", ParallelExecutionAlgorithm),
]

N_ITERATIONS = 5  # Run each benchmark multiple times for stable timing


@dataclass
class BenchmarkResult:
    name: str
    algorithm: str
    time_s: float
    memory_mb: float
    correct: bool
    iterations: int


def compare_output(actual: dict, expected: dict) -> bool:
    """Compare workflow output to expected values, tolerating Value wrappers."""
    for key, expected_val in expected.items():
        if key not in actual:
            return False
        actual_val = actual[key]
        # Unwrap Value objects
        if hasattr(actual_val, "root"):
            actual_val = actual_val.root
        if hasattr(expected_val, "root"):
            expected_val = expected_val.root
        # Float comparison with tolerance
        if isinstance(actual_val, float) and isinstance(expected_val, (int, float)):
            if abs(actual_val - expected_val) > 1e-9:
                return False
        elif actual_val != expected_val:
            return False
    return True


async def run_benchmark(
    name: str,
    workflow: Workflow,
    input_data: dict,
    expected: dict,
    algorithm_name: str,
    algorithm_cls: type,
) -> BenchmarkResult:
    """Run a single benchmark N_ITERATIONS times and return aggregate results."""
    context = InMemoryContext()
    algorithm = algorithm_cls()

    # Warmup run
    await algorithm.execute(context=context, workflow=workflow, input=input_data)

    # Timed runs
    tracemalloc.start()
    tracemalloc.reset_peak()

    start = time.perf_counter()
    correct = True
    for _ in range(N_ITERATIONS):
        result = await algorithm.execute(
            context=context, workflow=workflow, input=input_data
        )
        if result.status is not WorkflowExecutionResultStatus.SUCCESS:
            correct = False
        elif not compare_output(result.output, expected):
            correct = False
    elapsed = time.perf_counter() - start

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name=name,
        algorithm=algorithm_name,
        time_s=elapsed,
        memory_mb=peak_bytes / (1024 * 1024),
        correct=correct,
        iterations=N_ITERATIONS,
    )


async def main():
    results: list[BenchmarkResult] = []

    for bench_name, builder in BENCHMARKS:
        workflow, input_data, expected = builder()
        for algo_name, algo_cls in ALGORITHMS:
            r = await run_benchmark(
                bench_name, workflow, input_data, expected, algo_name, algo_cls
            )
            results.append(r)

    # Print grep-able summary
    total_time = sum(r.time_s for r in results)
    peak_memory = max(r.memory_mb for r in results)
    correct_count = sum(1 for r in results if r.correct)
    total_count = len(results)

    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Benchmark':<20} {'Algorithm':<15} {'Time (s)':<12} {'Memory (MB)':<14} {'Correct'}")
    print("-" * 70)
    for r in results:
        print(
            f"{r.name:<20} {r.algorithm:<15} {r.time_s:<12.4f} {r.memory_mb:<14.2f} {'PASS' if r.correct else 'FAIL'}"
        )
    print("-" * 70)
    print()

    # Grep-able metrics (these are what program.md tells the agent to extract)
    print(f"total_time_s: {total_time:.4f}")
    print(f"peak_memory_mb: {peak_memory:.2f}")
    print(f"correctness: {correct_count}/{total_count}")
    print(f"iterations_per_benchmark: {N_ITERATIONS}")


if __name__ == "__main__":
    asyncio.run(main())
