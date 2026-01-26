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
    Executes workflow nodes in parallel using asyncio with eager dispatch.

    Nodes are dispatched as soon as their dependencies are satisfied, without
    waiting for other concurrent nodes to complete. This maximizes parallelism
    when nodes have varying execution times.

    When a node expands into a sub-workflow, the expansion is processed and
    newly ready nodes are dispatched immediately.

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
        running_tasks: dict[asyncio.Task[NodeResult], str] = {}  # task -> node_id
        errors = WorkflowErrors()

        try:
            # Initial dispatch - start all initially ready nodes
            ready_nodes = workflow.get_ready_nodes(input=input)
            for node_id, node_input in ready_nodes.items():
                task = asyncio.create_task(
                    self._execute_node(context, workflow, node_id, node_input, semaphore)
                )
                running_tasks[task] = node_id

            # Main event loop - process completions eagerly
            while running_tasks:
                done, _ = await asyncio.wait(
                    running_tasks.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Process completed tasks
                expansions_pending: list[tuple[str, Workflow]] = []

                for task in done:
                    node_id = running_tasks.pop(task)

                    try:
                        node_result = task.result()
                    except Exception as e:
                        if self.error_handling == ErrorHandlingMode.FAIL_FAST:
                            await self._cancel_all(running_tasks)
                            raise
                        errors.add(e)
                        failed_nodes.add(node_id)
                        continue

                    if isinstance(node_result.result, Workflow):
                        expansions_pending.append((node_id, node_result.result))
                    elif isinstance(node_result.result, Exception):
                        # Handle exception stored in NodeResult (shouldn't happen with current _execute_node)
                        if self.error_handling == ErrorHandlingMode.FAIL_FAST:
                            await self._cancel_all(running_tasks)
                            raise node_result.result
                        errors.add(node_result.result)
                        failed_nodes.add(node_id)
                    else:
                        node_outputs[node_id] = node_result.result

                # Process expansions sequentially (workflow is immutable)
                for node_id, subgraph in expansions_pending:
                    workflow = workflow.expand_node(node_id, subgraph)

                # EAGERLY dispatch newly ready nodes
                in_flight = set(running_tasks.values())
                ready_nodes = {
                    nid: inp
                    for nid, inp in workflow.get_ready_nodes(
                        input=input,
                        node_outputs=node_outputs,
                    ).items()
                    if nid not in failed_nodes and nid not in in_flight
                }

                for node_id, node_input in ready_nodes.items():
                    task = asyncio.create_task(
                        self._execute_node(context, workflow, node_id, node_input, semaphore)
                    )
                    running_tasks[task] = node_id

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

    async def _cancel_all(
        self, running_tasks: dict[asyncio.Task[NodeResult], str]
    ) -> None:
        """Cancel all running tasks and wait for completion."""
        for task in running_tasks:
            task.cancel()
        if running_tasks:
            await asyncio.wait(running_tasks.keys(), return_when=asyncio.ALL_COMPLETED)

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
