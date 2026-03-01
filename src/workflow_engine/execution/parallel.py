# workflow_engine/execution/parallel.py

import asyncio
from collections.abc import Mapping
from enum import StrEnum
from typing import NamedTuple

from overrides import override

from ..core import (
    Context,
    DataMapping,
    ExecutionAlgorithm,
    Node,
    NodeException,
    ShouldRetry,
    ShouldYield,
    Workflow,
    WorkflowErrors,
    WorkflowExecutionResult,
)
from .rate_limit import RateLimitRegistry
from .retry import RetryTracker


class ErrorHandlingMode(StrEnum):
    """Configures how the parallel executor handles node errors."""

    FAIL_FAST = "fail_fast"  # Stop on first error
    CONTINUE = "continue"  # Continue execution, collect all errors


class NodeResult(NamedTuple):
    """Result of a single node execution."""

    node_id: str
    result: DataMapping | Workflow | Exception
    input: DataMapping  # Original input to the node
    should_retry: ShouldRetry | None = None  # Set if this is a retryable failure
    should_yield: ShouldYield | None = None  # Set if the node yielded


class ParallelExecutionAlgorithm(ExecutionAlgorithm):
    """
    Executes workflow nodes in parallel using asyncio with eager dispatch.

    Nodes are dispatched as soon as their dependencies are satisfied, without
    waiting for other concurrent nodes to complete. This maximizes parallelism
    when nodes have varying execution times.

    When a node expands into a sub-workflow, the expansion is processed and
    newly ready nodes are dispatched immediately.

    Supports retry with backoff for transient failures (ShouldRetry exceptions)
    and rate limiting per node type.

    Args:
        error_handling: How to handle node errors (default: FAIL_FAST)
        max_concurrency: Maximum number of concurrent nodes (default: None = unlimited)
        max_retries: Maximum retry attempts for nodes (default: 3)
        rate_limits: Registry of rate limit configurations per node type
    """

    def __init__(
        self,
        error_handling: ErrorHandlingMode = ErrorHandlingMode.FAIL_FAST,
        max_concurrency: int | None = None,
        max_retries: int = 3,
        rate_limits: RateLimitRegistry | None = None,
    ):
        self.error_handling = error_handling
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.rate_limits = rate_limits or RateLimitRegistry()

    def _get_node_max_retries(self, node: Node) -> int | None:
        """Get the max retries for a node, checking NodeTypeInfo first."""
        if node.TYPE_INFO.max_retries is not None:
            return node.TYPE_INFO.max_retries
        return None

    @override
    async def execute(
        self,
        *,
        context: Context,
        workflow: Workflow,
        input: DataMapping,
    ) -> WorkflowExecutionResult:
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
        retry_tracker = RetryTracker(default_max_retries=self.max_retries)

        # Track nodes that are waiting for retry (node_id -> input)
        pending_retry: dict[str, DataMapping] = {}
        # Track nodes that yielded (node_id -> ShouldYield message)
        node_yields: dict[str, str] = {}

        try:
            # Initial dispatch - start all initially ready nodes
            ready_nodes = {workflow.input_node.id: input}
            for node_id, node_input in ready_nodes.items():
                task = asyncio.create_task(
                    self._execute_node(
                        context, workflow, node_id, node_input, semaphore, retry_tracker
                    )
                )
                running_tasks[task] = node_id

            # Main event loop - process completions eagerly
            while running_tasks or pending_retry:
                # Check if any pending retries are now ready
                for node_id in list(pending_retry.keys()):
                    state = retry_tracker.get_state(node_id)
                    if state.is_ready():
                        node_input = pending_retry.pop(node_id)
                        task = asyncio.create_task(
                            self._execute_node(
                                context,
                                workflow,
                                node_id,
                                node_input,
                                semaphore,
                                retry_tracker,
                            )
                        )
                        running_tasks[task] = node_id

                # If no tasks are running but retries are pending, wait for shortest backoff
                if not running_tasks and pending_retry:
                    wait_time = retry_tracker.min_wait_time()
                    if wait_time and wait_time.total_seconds() > 0:
                        await asyncio.sleep(wait_time.total_seconds())
                    continue

                if not running_tasks:
                    break

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

                    # Handle yielded nodes
                    if node_result.should_yield is not None:
                        node = workflow.nodes_by_id[node_id]
                        node_yields[node_id] = node_result.should_yield.message
                        await context.on_node_yield(
                            node=node,
                            input=node_result.input,
                            exception=node_result.should_yield,
                        )
                        continue

                    # Handle retryable failures
                    if node_result.should_retry is not None:
                        should_retry_error = node_result.should_retry
                        node = workflow.nodes_by_id[node_id]
                        node_max_retries = self._get_node_max_retries(node)
                        node_input = node_result.input

                        if retry_tracker.should_retry(node_id, node_max_retries):
                            retry_tracker.record_retry(node_id, should_retry_error)
                            pending_retry[node_id] = node_input

                            # Call the on_node_retry hook
                            state = retry_tracker.get_state(node_id)
                            await context.on_node_retry(
                                node=node,
                                input=node_input,
                                exception=should_retry_error,
                                attempt=state.attempt,
                            )
                            continue

                        # Max retries exceeded - treat as failure
                        if self.error_handling == ErrorHandlingMode.FAIL_FAST:
                            await self._cancel_all(running_tasks)
                            raise should_retry_error
                        errors.add(should_retry_error)
                        failed_nodes.add(node_id)
                        continue

                    if isinstance(node_result.result, Workflow):
                        expansions_pending.append((node_id, node_result.result))
                    elif isinstance(node_result.result, Exception):
                        # Handle exception stored in NodeResult
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
                pending_set = set(pending_retry.keys())
                ready_nodes = {
                    nid: inp
                    for nid, inp in workflow.get_ready_nodes(
                        node_outputs=node_outputs,
                    ).items()
                    if nid not in failed_nodes
                    and nid not in in_flight
                    and nid not in pending_set
                    and nid not in node_yields
                }

                for node_id, node_input in ready_nodes.items():
                    task = asyncio.create_task(
                        self._execute_node(
                            context,
                            workflow,
                            node_id,
                            node_input,
                            semaphore,
                            retry_tracker,
                        )
                    )
                    running_tasks[task] = node_id

            # Short-circuit before attempting full output if errors were
            # collected in CONTINUE mode, to avoid masking real errors with
            # spurious "missing output" exceptions from yielded/failed nodes.
            if errors.any():
                partial_output = await workflow.get_output(
                    context=context,
                    node_outputs=node_outputs,
                    partial=True,
                )
                result = await context.on_workflow_error(
                    workflow=workflow,
                    input=input,
                    errors=errors,
                    partial_output=partial_output,
                    node_yields=node_yields,
                )
                return result

            if len(node_yields) > 0:
                partial_output = await workflow.get_output(
                    context=context,
                    node_outputs=node_outputs,
                    partial=True,
                )
                result = await context.on_workflow_yield(
                    workflow=workflow,
                    input=input,
                    partial_output=partial_output,
                    node_yields=node_yields,
                )
                return result

            output = await workflow.get_output(
                context=context,
                node_outputs=node_outputs,
            )
        except Exception as e:
            errors.add(e)
            partial_output = await workflow.get_output(
                context=context,
                node_outputs=node_outputs,
                partial=True,
            )
            result = await context.on_workflow_error(
                workflow=workflow,
                input=input,
                errors=errors,
                partial_output=partial_output,
                node_yields=node_yields,
            )
            return result
        else:
            result = await context.on_workflow_finish(
                workflow=workflow,
                input=input,
                output=output,
            )
            return result

    async def _cancel_all(
        self,
        running_tasks: Mapping[asyncio.Task[NodeResult], str],
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
        retry_tracker: RetryTracker,
    ) -> NodeResult:
        """Execute a single node with rate limiting and retry support."""
        node = workflow.nodes_by_id[node_id]

        # Acquire rate limiter if configured for this node type
        limiter = self.rate_limits.get_limiter(node.type)
        if limiter is not None:
            await limiter.acquire()

        try:
            if semaphore is not None:
                async with semaphore:
                    result = await node(context, node_input)
            else:
                result = await node(context, node_input)

            return NodeResult(node_id, result, input=node_input)

        except ShouldYield as e:
            return NodeResult(
                node_id, result=node_input, input=node_input, should_yield=e
            )

        except NodeException as e:
            # Check if the underlying cause is ShouldRetry
            if isinstance(e.__cause__, ShouldRetry):
                return NodeResult(
                    node_id,
                    result=node_input,
                    input=node_input,
                    should_retry=e.__cause__,
                )
            raise

        finally:
            if limiter is not None:
                limiter.release()


__all__ = [
    "ErrorHandlingMode",
    "ParallelExecutionAlgorithm",
]
