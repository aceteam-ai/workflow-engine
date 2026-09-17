"""Compare direct replacement with a one-node Workflow wrapper (no I/O).

Run with ``PYTHONPATH=src python scripts/benchmark_node_replacement.py``.
Timings include public engine validation and default no-op checkpoint hooks;
this is a reproducible observation, not a speed or same-worker guarantee.
"""

import argparse
import asyncio
import json
import statistics
import time
from typing import ClassVar

from overrides import override
from pydantic import Field

from workflow_engine import (
    Data,
    DataMapping,
    Edge,
    IntegerValue,
    Node,
    NodeTypeInfo,
    Params,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.execution.parallel import ParallelExecutionAlgorithm


class BenchmarkData(Data):
    value: IntegerValue = Field(title="Value", description="The unchanged payload.")


class BenchmarkLeafNode(Node[BenchmarkData, BenchmarkData, Params]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Replacement benchmark leaf",
        version="1.0.0",
        parameter_type=Params,
    )

    @classmethod
    @override
    def static_input_type(cls) -> type[BenchmarkData]:
        return BenchmarkData

    @classmethod
    @override
    def static_output_type(cls) -> type[BenchmarkData]:
        return BenchmarkData

    @override
    async def run(
        self, *, context, input_type, output_type, input
    ) -> BenchmarkData | Node | Workflow:
        return input


class BenchmarkDirectNode(BenchmarkLeafNode):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Direct benchmark", version="1.0.0", parameter_type=Params
    )

    @override
    async def run(self, *, context, input_type, output_type, input) -> Node:
        return BenchmarkLeafNode(type="BenchmarkLeaf", id="leaf", params=Params())


class BenchmarkWrappedNode(BenchmarkLeafNode):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Wrapped benchmark", version="1.0.0", parameter_type=Params
    )

    @override
    async def run(self, *, context, input_type, output_type, input) -> Workflow:
        engine = WorkflowEngine()
        return Workflow(
            input_node=engine.create_input_node(value=IntegerValue),
            output_node=engine.create_output_node(value=IntegerValue),
            inner_nodes=[engine.create_node(BenchmarkLeafNode, id="leaf")],
            edges=[
                Edge(
                    source_id="input",
                    source_key="value",
                    target_id="leaf",
                    target_key="value",
                ),
                Edge(
                    source_id="leaf",
                    source_key="value",
                    target_id="output",
                    target_key="value",
                ),
            ],
        )


class CountingContext(InMemoryExecutionContext):
    def __init__(self):
        super().__init__()
        self.starts = 0
        self.finishes = 0

    @override
    async def on_node_start(
        self, *, node, input_type, output_type, input
    ) -> DataMapping | Workflow | Node | None:
        self.starts += 1
        return None

    @override
    async def on_node_finish(
        self, *, node, input_type, output_type, input, output
    ) -> DataMapping:
        self.finishes += 1
        return output


async def benchmark(iterations: int, warmup: int) -> None:
    for algorithm in (TopologicalExecutionAlgorithm(), ParallelExecutionAlgorithm()):
        engine = WorkflowEngine(execution_algorithm=algorithm)
        for node in (BenchmarkDirectNode, BenchmarkWrappedNode):
            graph = await engine.build_single_node_workflow(node)
            elapsed: list[float] = []
            counts: set[tuple[int, int]] = set()
            for iteration in range(iterations + warmup):
                context = CountingContext()
                started = time.perf_counter()
                result = await engine.execute(
                    context=context, workflow=graph, input={"value": 7}
                )
                duration = time.perf_counter() - started
                assert result.status is WorkflowExecutionResultStatus.SUCCESS, (
                    result.errors
                )
                assert result.output["value"] == IntegerValue(7)
                if iteration >= warmup:
                    elapsed.append(duration * 1000)
                    counts.add((context.starts, context.finishes))
            assert len(counts) == 1, counts
            starts, finishes = counts.pop()
            print(
                json.dumps(
                    {
                        "algorithm": type(algorithm).__name__,
                        "mode": "direct" if node is BenchmarkDirectNode else "wrapped",
                        "iterations": iterations,
                        "median_ms": round(statistics.median(elapsed), 3),
                        "start_hooks": starts,
                        "finish_hooks": finishes,
                    }
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    asyncio.run(benchmark(args.iterations, args.warmup))
