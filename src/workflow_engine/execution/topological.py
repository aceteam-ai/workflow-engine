# workflow_engine/execution/topological.py
"""
Topological execution algorithm with retry and rate limiting support.
"""

import asyncio

from overrides import override

from ..core import Context, DataMapping, ExecutionAlgorithm, Workflow, WorkflowErrors
from ..core.error import NodeException, ShouldRetry
from .rate_limit import RateLimitRegistry
from .retry import RetryTracker


class TopologicalExecutionAlgorithm(ExecutionAlgorithm):
    """
    Executes the workflow one node at a time on the current thread, in
    topological order.

    Supports retry with backoff for transient failures (ShouldRetry exceptions)
    and rate limiting per node type.
    """

    def __init__(
        self,
        max_retries: int = 3,
        rate_limits: RateLimitRegistry | None = None,
    ):
        """
        Initialize the execution algorithm.

        max_retries: default maximum retry attempts for nodes (can be overridden
                     per node type via NodeTypeInfo.max_retries)
        rate_limits: registry of rate limit configurations per node type
        """
        self.max_retries = max_retries
        self.rate_limits = rate_limits or RateLimitRegistry()

    def _get_node_max_retries(self, node) -> int | None:
        """Get the max retries for a node, checking NodeTypeInfo first."""
        if hasattr(node, "TYPE_INFO") and node.TYPE_INFO.max_retries is not None:
            return node.TYPE_INFO.max_retries
        return None

    @override
    async def execute(
        self,
        *,
        context: Context,
        workflow: Workflow,
        input: DataMapping,
    ) -> tuple[WorkflowErrors, DataMapping]:
        result = await context.on_workflow_start(workflow=workflow, input=input)
        if result is not None:
            return result

        node_outputs: dict[str, DataMapping] = {}
        errors = WorkflowErrors()
        retry_tracker = RetryTracker(default_max_retries=self.max_retries)

        # Track nodes that are waiting for retry (node_id -> input)
        pending_retry: dict[str, DataMapping] = {}

        try:
            ready_nodes = dict(workflow.get_ready_nodes(input=input))

            while ready_nodes or pending_retry:
                # Check if any pending retries are now ready
                for node_id in list(pending_retry.keys()):
                    state = retry_tracker.get_state(node_id)
                    if state.is_ready():
                        ready_nodes[node_id] = pending_retry.pop(node_id)

                # If no nodes are ready, wait for the shortest backoff
                if not ready_nodes and pending_retry:
                    wait_time = retry_tracker.min_wait_time()
                    if wait_time and wait_time.total_seconds() > 0:
                        await asyncio.sleep(wait_time.total_seconds())
                    continue

                if not ready_nodes:
                    break

                node_id, node_input = ready_nodes.popitem()
                node = workflow.nodes_by_id[node_id]

                # Acquire rate limiter if configured for this node type
                limiter = self.rate_limits.get_limiter(node.type)
                if limiter is not None:
                    await limiter.acquire()

                try:
                    node_result = await node(context, node_input)

                    if isinstance(node_result, Workflow):
                        workflow = workflow.expand_node(node_id, node_result)
                    else:
                        node_outputs[node.id] = node_result

                except NodeException as e:
                    # Check if the underlying cause is ShouldRetry
                    if isinstance(e.__cause__, ShouldRetry):
                        should_retry_error = e.__cause__
                        node_max_retries = self._get_node_max_retries(node)

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

                            # Continue to next node without re-raising
                            continue

                    # Max retries exceeded or non-retryable error
                    raise

                finally:
                    if limiter is not None:
                        limiter.release()

                ready_nodes = dict(
                    workflow.get_ready_nodes(
                        input=input,
                        node_outputs=node_outputs,
                        partial_results=ready_nodes,
                    )
                )

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

        output = await context.on_workflow_finish(
            workflow=workflow,
            input=input,
            output=output,
        )

        return errors, output


__all__ = [
    "TopologicalExecutionAlgorithm",
]
