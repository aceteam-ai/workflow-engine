#!/usr/bin/env python3
"""
Autoresearch benchmark harness (READ-ONLY — do not modify).

Builds and executes 8 benchmark workflows programmatically, measuring:
  - Wall-clock execution time (seconds)
  - Peak memory usage (MB)
  - Correctness (output matches expected values)

Benchmarks cover:
  - Simple arithmetic DAGs (linear, branching, nested, fan-out/fan-in)
  - Large node counts (100-node DAG)
  - Sub-workflow expansion via ForEachNode
  - Retry with backoff (ShouldRetry)
  - Yield/resume (ShouldYield)

All workflows use deterministic nodes — no LLM calls, no API keys.

Usage:
    uv run python autoresearch/evaluate.py
"""

import asyncio
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Literal

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engine import (
    AddNode,
    ConstantIntegerNode,
    ConstantStringNode,
    Context,
    Data,
    DataValue,
    Edge,
    Empty,
    FloatValue,
    ForEachNode,
    InMemoryContext,
    InputNode,
    IntegerValue,
    Node,
    NodeTypeInfo,
    OutputNode,
    Params,
    ParallelExecutionAlgorithm,
    SequenceValue,
    ShouldRetry,
    ShouldYield,
    StringValue,
    SumNode,
    TopologicalExecutionAlgorithm,
    Workflow,
    WorkflowExecutionResultStatus,
    WorkflowValue,
    get_data_dict,
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
# Custom nodes for advanced benchmarks
# ---------------------------------------------------------------------------


class BenchRetryInput(Data):
    value: StringValue


class BenchRetryOutput(Data):
    result: StringValue


class BenchRetryParams(Params):
    fail_count: int


class BenchRetryNode(Node[BenchRetryInput, BenchRetryOutput, BenchRetryParams]):
    """Fails fail_count times with ShouldRetry, then succeeds."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="BenchRetry",
        display_name="Bench Retry",
        description="Benchmark retry node.",
        version="1.0.0",
        parameter_type=BenchRetryParams,
    )

    type: Literal["BenchRetry"] = "BenchRetry"  # pyright: ignore[reportIncompatibleVariableOverride]
    _attempt_counts: ClassVar[dict[str, int]] = {}

    @cached_property
    def input_type(self):
        return BenchRetryInput

    @cached_property
    def output_type(self):
        return BenchRetryOutput

    async def run(self, context: Context, input: BenchRetryInput) -> BenchRetryOutput:
        if self.id not in BenchRetryNode._attempt_counts:
            BenchRetryNode._attempt_counts[self.id] = 0
        BenchRetryNode._attempt_counts[self.id] += 1

        if BenchRetryNode._attempt_counts[self.id] <= self.params.fail_count:
            raise ShouldRetry(
                message=f"Transient failure (attempt {BenchRetryNode._attempt_counts[self.id]})",
                backoff=timedelta(milliseconds=1),  # Minimal backoff for benchmarks
            )
        return BenchRetryOutput(result=StringValue(f"ok:{input.value.root}"))

    @classmethod
    def from_fail_count(cls, id: str, fail_count: int) -> "BenchRetryNode":
        return cls(id=id, params=BenchRetryParams(fail_count=fail_count))

    @classmethod
    def reset(cls):
        cls._attempt_counts = {}


class BenchYieldInput(Data):
    value: StringValue


class BenchYieldOutput(Data):
    result: StringValue


class BenchYieldNode(Node[BenchYieldInput, BenchYieldOutput, Params]):
    """Yields on first N calls, succeeds after that. Simulates async dispatch."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="BenchYield",
        display_name="Bench Yield",
        description="Benchmark yield node.",
        version="1.0.0",
        parameter_type=Params,
    )

    type: Literal["BenchYield"] = "BenchYield"  # pyright: ignore[reportIncompatibleVariableOverride]
    _call_counts: ClassVar[dict[str, int]] = {}
    yield_count: int = 1  # Number of times to yield before succeeding

    @cached_property
    def input_type(self):
        return BenchYieldInput

    @cached_property
    def output_type(self):
        return BenchYieldOutput

    async def run(self, context: Context, input: BenchYieldInput) -> BenchYieldOutput:
        n = BenchYieldNode._call_counts.get(self.id, 0) + 1
        BenchYieldNode._call_counts[self.id] = n
        if n <= self.yield_count:
            raise ShouldYield(f"waiting ({n}/{self.yield_count})")
        return BenchYieldOutput(result=StringValue(f"resumed:{input.value.root}"))

    @classmethod
    def reset(cls):
        cls._call_counts = {}


# ---------------------------------------------------------------------------
# Advanced benchmark workflow builders
# ---------------------------------------------------------------------------


def build_large_100() -> tuple[Workflow, dict, dict]:
    """100-node DAG: 10 parallel chains of 10 sequential additions.

    Each chain: const(chain_idx * 100) → add(+1) → add(+1) → ... (9 adds)
    Final: sum all 10 chain outputs via AddNode(arity=10)

    Chain i output = chain_idx * 100 + 9
    Expected: sum((i * 100 + 9) for i in range(10)) = 4590

    Nodes: 10 seed-constants + 90 add-constants + 90 adds + 1 final + input + output = 193
    """
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(result=IntegerValue)

    n_chains = 10
    chain_length = 9  # 9 additions per chain
    inner_nodes = []
    edges = []
    chain_tails = []

    for chain_idx in range(n_chains):
        seed = ConstantIntegerNode.from_value(
            id=f"seed_{chain_idx}", value=chain_idx * 100
        )
        inner_nodes.append(seed)
        prev_source = seed
        prev_key = "value"

        for step in range(chain_length):
            step_const = ConstantIntegerNode.from_value(
                id=f"ch{chain_idx}_c{step}", value=1
            )
            step_add = AddNode(id=f"ch{chain_idx}_add{step}")
            inner_nodes.extend([step_const, step_add])
            edges.append(
                Edge.from_nodes(
                    source=prev_source,
                    source_key=prev_key,
                    target=step_add,
                    target_key="a",
                )
            )
            edges.append(
                Edge.from_nodes(
                    source=step_const,
                    source_key="value",
                    target=step_add,
                    target_key="b",
                )
            )
            prev_source = step_add
            prev_key = "sum"

        chain_tails.append((prev_source, prev_key))

    final_add = AddNode.with_arity(id="final_sum", arity=n_chains)
    inner_nodes.append(final_add)

    field_names = [chr(ord("a") + i) for i in range(n_chains)]
    for i, (tail_node, tail_key) in enumerate(chain_tails):
        edges.append(
            Edge.from_nodes(
                source=tail_node,
                source_key=tail_key,
                target=final_add,
                target_key=field_names[i],
            )
        )
    edges.append(
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
        inner_nodes=inner_nodes,
        edges=edges,
    )

    input_data = {}
    expected = {"result": sum(i * 100 + chain_length for i in range(n_chains))}
    return workflow, input_data, expected


def build_foreach_expansion() -> tuple[Workflow, dict, dict]:
    """ForEach sub-workflow expansion: iterate over 6 items, doubling each.

    Inner workflow: input(x) → add(x + x) → output(y)
    ForEach: [1, 2, 3, 4, 5, 6] → [2, 4, 6, 8, 10, 12]

    This exercises expand_node() / graph surgery at runtime.
    """
    # Inner workflow: doubles input
    inner_input = InputNode.from_fields(x=FloatValue)
    inner_output = OutputNode.from_fields(y=FloatValue)
    inner_add = AddNode(id="double")

    inner_workflow = Workflow(
        input_node=inner_input,
        output_node=inner_output,
        inner_nodes=[inner_add],
        edges=[
            Edge.from_nodes(
                source=inner_input, source_key="x", target=inner_add, target_key="a"
            ),
            Edge.from_nodes(
                source=inner_input, source_key="x", target=inner_add, target_key="b"
            ),
            Edge.from_nodes(
                source=inner_add, source_key="sum", target=inner_output, target_key="y"
            ),
        ],
    )

    # Outer workflow with ForEach
    foreach = ForEachNode.from_workflow(id="foreach", workflow=inner_workflow)
    outer_input = InputNode.from_fields(sequence=SequenceValue[FloatValue])
    outer_output = OutputNode.from_fields(results=SequenceValue[FloatValue])

    workflow = Workflow(
        input_node=outer_input,
        output_node=outer_output,
        inner_nodes=[foreach],
        edges=[
            Edge.from_nodes(
                source=outer_input,
                source_key="sequence",
                target=foreach,
                target_key="sequence",
            ),
            Edge.from_nodes(
                source=foreach,
                source_key="sequence",
                target=outer_output,
                target_key="results",
            ),
        ],
    )

    items = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    input_data = get_data_dict(
        workflow.input_type.model_validate({"sequence": items})
    )
    expected = {"results": [v * 2 for v in items]}
    return workflow, input_data, expected


def build_retry_chain() -> tuple[Workflow, dict, dict]:
    """Chain of 3 retryable nodes, each failing once before succeeding.

    const("start") → retry1(fail=1) → retry2(fail=1) → retry3(fail=1) → output
    Expected: "ok:ok:ok:start"

    Exercises the retry/backoff machinery in the execution loop.
    """
    BenchRetryNode.reset()

    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(result=StringValue)

    constant = ConstantStringNode.from_value(id="const_start", value="start")
    r1 = BenchRetryNode.from_fail_count(id="bench_retry_1", fail_count=1)
    r2 = BenchRetryNode.from_fail_count(id="bench_retry_2", fail_count=1)
    r3 = BenchRetryNode.from_fail_count(id="bench_retry_3", fail_count=1)

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[constant, r1, r2, r3],
        edges=[
            Edge.from_nodes(
                source=constant, source_key="value", target=r1, target_key="value"
            ),
            Edge.from_nodes(
                source=r1, source_key="result", target=r2, target_key="value"
            ),
            Edge.from_nodes(
                source=r2, source_key="result", target=r3, target_key="value"
            ),
            Edge.from_nodes(
                source=r3, source_key="result", target=output_node, target_key="result"
            ),
        ],
    )

    input_data = {}
    expected = {"result": "ok:ok:ok:start"}
    return workflow, input_data, expected


def build_yield_resume() -> tuple[Workflow, dict, dict]:
    """Workflow with 2 parallel branches: one yields, one succeeds immediately.

    const("hello") → yield_node (yields once, needs re-run to succeed)
                   → echo_node (succeeds immediately)
    Output: both branches' results

    Tests yield handling and partial progress. We run it twice:
    first run yields, second run succeeds. The benchmark checks that
    re-running after yield produces the correct final output.
    """
    BenchYieldNode.reset()

    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(
        yielded=StringValue,
        immediate=StringValue,
    )

    constant = ConstantStringNode.from_value(id="const_hello", value="hello")

    # Echo node that just passes through (use a constant + add trick: add "hello" + const("") won't work
    # with string types. Let's use two constant string nodes instead.
    # Actually, we need something simpler — just wire const directly to output for the "immediate" path,
    # and have the yield node on the other path.
    yield_node = BenchYieldNode(id="bench_yield_node", params=Params(), yield_count=1)

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[constant, yield_node],
        edges=[
            Edge.from_nodes(
                source=constant, source_key="value", target=yield_node, target_key="value"
            ),
            Edge.from_nodes(
                source=yield_node,
                source_key="result",
                target=output_node,
                target_key="yielded",
            ),
            Edge.from_nodes(
                source=constant,
                source_key="value",
                target=output_node,
                target_key="immediate",
            ),
        ],
    )

    input_data = {}
    expected = {"yielded": "resumed:hello", "immediate": "hello"}
    return workflow, input_data, expected


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

BENCHMARKS = [
    ("linear_5", build_linear_5),
    ("branching_10", build_branching_10),
    ("nested_20", build_nested_20),
    ("parallel_fan", build_parallel_fan),
    ("large_100", build_large_100),
    ("foreach_expand", build_foreach_expansion),
    ("retry_chain", build_retry_chain),
    ("yield_resume", build_yield_resume),
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


def _unwrap(val):
    """Recursively unwrap Value objects to plain Python types."""
    if hasattr(val, "root"):
        val = val.root
    if isinstance(val, list):
        return [_unwrap(v) for v in val]
    return val


def compare_output(actual: dict, expected: dict) -> bool:
    """Compare workflow output to expected values, tolerating Value wrappers."""
    for key, expected_val in expected.items():
        if key not in actual:
            return False
        actual_val = _unwrap(actual[key])
        expected_val = _unwrap(expected_val)
        # List comparison (e.g., SequenceValue results)
        if isinstance(expected_val, list) and isinstance(actual_val, list):
            if len(actual_val) != len(expected_val):
                return False
            for a, e in zip(actual_val, expected_val):
                a = _unwrap(a)
                e = _unwrap(e)
                if isinstance(a, float) and isinstance(e, (int, float)):
                    if abs(a - e) > 1e-9:
                        return False
                elif a != e:
                    return False
            continue
        # Float comparison with tolerance
        if isinstance(actual_val, float) and isinstance(expected_val, (int, float)):
            if abs(actual_val - expected_val) > 1e-9:
                return False
        elif actual_val != expected_val:
            return False
    return True


def _reset_bench_nodes():
    """Reset stateful benchmark nodes between iterations."""
    BenchRetryNode.reset()
    BenchYieldNode.reset()


async def _run_yield_benchmark(
    algorithm,
    context: InMemoryContext,
    workflow: Workflow,
    input_data: dict,
    expected: dict,
) -> bool:
    """Run a yield benchmark: first run yields, second run succeeds."""
    _reset_bench_nodes()

    # First execution — should yield
    result = await algorithm.execute(context=context, workflow=workflow, input=input_data)
    if result.status is not WorkflowExecutionResultStatus.YIELDED:
        return False

    # Second execution — should succeed after resume
    result = await algorithm.execute(context=context, workflow=workflow, input=input_data)
    if result.status is not WorkflowExecutionResultStatus.SUCCESS:
        return False
    return compare_output(result.output, expected)


async def _run_retry_benchmark(
    algorithm,
    context: InMemoryContext,
    workflow: Workflow,
    input_data: dict,
    expected: dict,
) -> bool:
    """Run a retry benchmark: nodes fail then succeed within retry limit."""
    _reset_bench_nodes()

    result = await algorithm.execute(context=context, workflow=workflow, input=input_data)
    if result.status is not WorkflowExecutionResultStatus.SUCCESS:
        return False
    return compare_output(result.output, expected)


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

    is_yield = name == "yield_resume"
    is_retry = name == "retry_chain"

    # Warmup run
    if is_yield:
        _reset_bench_nodes()
        await algorithm.execute(context=context, workflow=workflow, input=input_data)
        await algorithm.execute(context=context, workflow=workflow, input=input_data)
    elif is_retry:
        _reset_bench_nodes()
        await algorithm.execute(context=context, workflow=workflow, input=input_data)
    else:
        await algorithm.execute(context=context, workflow=workflow, input=input_data)

    # Timed runs
    tracemalloc.start()
    tracemalloc.reset_peak()

    start = time.perf_counter()
    correct = True
    for _ in range(N_ITERATIONS):
        if is_yield:
            if not await _run_yield_benchmark(
                algorithm, context, workflow, input_data, expected
            ):
                correct = False
        elif is_retry:
            if not await _run_retry_benchmark(
                algorithm, context, workflow, input_data, expected
            ):
                correct = False
        else:
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
