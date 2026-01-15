# workflow_engine/execution/parallel.py

import asyncio
from enum import Enum
from typing import NamedTuple

from overrides import override

from ..core import Context, DataMapping, ExecutionAlgorithm, Workflow, WorkflowErrors


class ErrorHandlingMode(Enum):
    """Configures how the parallel executor handles node errors."""

    FAIL_FAST = "fail_fast"  # Stop on first error
    CONTINUE = "continue"  # Continue execution, collect all errors


class NodeResult(NamedTuple):
    """Result of a single node execution."""

    node_id: str
    result: DataMapping | Workflow | Exception


class ParallelExecutionAlgorithm(ExecutionAlgorithm):
    """
    Executes workflow nodes in parallel using asyncio.

    All nodes whose dependencies are satisfied are executed concurrently.
    When a node expands into a sub-workflow, the algorithm completes the
    current batch, expands the node, and continues with the new graph.

    Args:
        error_handling: How to handle node errors (default: FAIL_FAST)
        max_concurrency: Maximum number of concurrent nodes (default: None = unlimited)
    """

    def __init__(
        self,
        error_handling: ErrorHandlingMode = ErrorHandlingMode.FAIL_FAST,
        max_concurrency: int | None = None,
    ):
        self.error_handling = error_handling
        self.max_concurrency = max_concurrency

    @override
    async def execute(
        self,
        *,
        context: Context,
        workflow: Workflow,
        input: DataMapping,
    ) -> tuple[WorkflowErrors, DataMapping]:
        # Initialize semaphore if max_concurrency is set
        semaphore: asyncio.Semaphore | None = None
        if self.max_concurrency is not None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        # Call workflow start hook
        result = await context.on_workflow_start(workflow=workflow, input=input)
        if result is not None:
            return result

        node_outputs: dict[str, DataMapping] = {}
        failed_nodes: set[str] = set()  # Track nodes that failed to avoid re-executing
        errors = WorkflowErrors()

        try:
            ready_nodes = dict(workflow.get_ready_nodes(input=input))

            while len(ready_nodes) > 0:
                # Execute all ready nodes in parallel
                results = await self._execute_batch(
                    context=context,
                    workflow=workflow,
                    ready_nodes=ready_nodes,
                    semaphore=semaphore,
                    errors=errors,
                )

                # Process results
                expansion_pending: list[tuple[str, Workflow]] = []

                for node_result in results:
                    if isinstance(node_result.result, Exception):
                        # Track failed node to avoid re-executing
                        failed_nodes.add(node_result.node_id)
                        continue
                    elif isinstance(node_result.result, Workflow):
                        # Queue expansion (handle after processing all results)
                        expansion_pending.append(
                            (node_result.node_id, node_result.result)
                        )
                    else:
                        # Normal output
                        node_outputs[node_result.node_id] = node_result.result

                # Handle node expansions
                for node_id, subgraph in expansion_pending:
                    workflow = workflow.expand_node(node_id, subgraph)

                # Get next batch of ready nodes, excluding failed ones
                ready_nodes = {
                    node_id: node_input
                    for node_id, node_input in workflow.get_ready_nodes(
                        input=input,
                        node_outputs=node_outputs,
                    ).items()
                    if node_id not in failed_nodes
                }

            output = workflow.get_output(node_outputs)

        except Exception as e:
            errors.add(e)
            partial_output = workflow.get_output(node_outputs, partial=True)
            errors, partial_output = await context.on_workflow_error(
                workflow=workflow,
                input=input,
                errors=errors,
                partial_output=partial_output,
            )
            return errors, partial_output

        # Check if we collected any errors in CONTINUE mode
        if errors.any():
            partial_output = workflow.get_output(node_outputs, partial=True)
            errors, partial_output = await context.on_workflow_error(
                workflow=workflow,
                input=input,
                errors=errors,
                partial_output=partial_output,
            )
            return errors, partial_output

        output = await context.on_workflow_finish(
            workflow=workflow,
            input=input,
            output=output,
        )

        return errors, output

    async def _execute_batch(
        self,
        context: Context,
        workflow: Workflow,
        ready_nodes: dict[str, DataMapping],
        semaphore: asyncio.Semaphore | None,
        errors: WorkflowErrors,
    ) -> list[NodeResult]:
        """Execute a batch of nodes in parallel."""
        tasks = [
            self._execute_node(context, workflow, node_id, node_input, semaphore)
            for node_id, node_input in ready_nodes.items()
        ]

        if self.error_handling == ErrorHandlingMode.FAIL_FAST:
            # First exception cancels all and propagates up
            return await asyncio.gather(*tasks)
        else:
            # CONTINUE mode: collect all results including exceptions
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            processed_results: list[NodeResult] = []
            node_ids = list(ready_nodes.keys())

            for i, raw_result in enumerate(raw_results):
                node_id = node_ids[i]
                if isinstance(raw_result, BaseException):
                    # BaseException includes SystemExit, KeyboardInterrupt, etc.
                    # Re-raise non-Exception BaseExceptions
                    if not isinstance(raw_result, Exception):
                        raise raw_result
                    errors.add(raw_result)
                    processed_results.append(NodeResult(node_id, raw_result))
                else:
                    # raw_result is NodeResult when no exception occurred
                    assert isinstance(raw_result, NodeResult)
                    processed_results.append(raw_result)

            return processed_results

    async def _execute_node(
        self,
        context: Context,
        workflow: Workflow,
        node_id: str,
        node_input: DataMapping,
        semaphore: asyncio.Semaphore | None,
    ) -> NodeResult:
        """Execute a single node, optionally with semaphore-based concurrency limiting."""
        node = workflow.nodes_by_id[node_id]

        if semaphore is not None:
            async with semaphore:
                result = await node(context, node_input)
        else:
            result = await node(context, node_input)

        return NodeResult(node_id, result)


__all__ = [
    "ErrorHandlingMode",
    "ParallelExecutionAlgorithm",
]
