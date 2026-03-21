# workflow_engine/execution/topological.py
"""
Topological execution algorithm with retry and rate limiting support.
"""

import asyncio

from overrides import override

from ..core import (
    Context,
    DataMapping,
    ExecutionAlgorithm,
    Workflow,
    WorkflowErrors,
    WorkflowExecutionResult,
)
from ..core.error import NodeException, ShouldRetry, ShouldYield
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

    @staticmethod
    def _build_successor_map(
        workflow: Workflow,
    ) -> dict[str, set[str]]:
        """Build a mapping from each node to the set of nodes that depend on it."""
        successors: dict[str, set[str]] = {node.id: set() for node in workflow.nodes}
        for edge in workflow.edges:
            successors[edge.source_id].add(edge.target_id)
        return successors

    @staticmethod
    def _check_node_ready(
        node_id: str,
        edges_by_target: dict[str, dict[str, "Edge"]],
        node_outputs: dict[str, DataMapping],
    ) -> DataMapping | None:
        """Check if a node is ready and return its input, or None if not ready."""
        node_edges = edges_by_target.get(node_id)
        if not node_edges:
            return {}
        node_input: dict[str, object] = {}
        for target_key, edge in node_edges.items():
            source_output = node_outputs.get(edge.source_id)
            if source_output is None:
                return None
            node_input[target_key] = source_output[edge.source_key]
        return node_input  # type: ignore

    @override
    async def execute(
        self,
        *,
        context: Context,
        workflow: Workflow,
        input: DataMapping,
    ) -> WorkflowExecutionResult:
        result = await context.on_workflow_start(workflow=workflow, input=input)
        if result is not None:
            return result

        node_outputs: dict[str, DataMapping] = {}
        errors = WorkflowErrors()
        retry_tracker = RetryTracker(default_max_retries=self.max_retries)

        # Track nodes that are waiting for retry (node_id -> input)
        pending_retry: dict[str, DataMapping] = {}
        # Track nodes that yielded (node_id -> yield message)
        node_yields: dict[str, str] = {}

        # Build successor map for incremental ready-node tracking
        successors = self._build_successor_map(workflow)
        edges_by_target = workflow.edges_by_target
        _check_ready = self._check_node_ready

        try:
            # Seed with ALL initially ready nodes (input node + nodes with no incoming edges)
            ready_nodes: dict[str, DataMapping] = {}
            for node in workflow.nodes:
                nid = node.id
                if nid == workflow.input_node.id:
                    ready_nodes[nid] = input
                elif not edges_by_target.get(nid):
                    ready_nodes[nid] = {}

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

                expanded = False
                try:
                    node_result = await node(context, node_input)

                    if isinstance(node_result, Workflow):
                        workflow = workflow.expand_node(node_id, node_result)
                        expanded = True
                    else:
                        node_outputs[node.id] = node_result

                except ShouldYield as e:
                    node_yields[node_id] = e.message
                    await context.on_node_yield(
                        node=node,
                        input=node_input,
                        exception=e,
                    )
                    continue

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

                if expanded:
                    # After expansion, rebuild maps and do a full scan
                    successors = self._build_successor_map(workflow)
                    edges_by_target = workflow.edges_by_target
                    ready_nodes = {
                        node_id: node_input
                        for node_id, node_input in workflow.get_ready_nodes(
                            node_outputs=node_outputs,
                            partial_results=ready_nodes,
                        ).items()
                        if node_id not in node_yields
                    }
                else:
                    # Incremental: only check successors of the just-completed node
                    for succ_id in successors.get(node_id, ()):
                        if succ_id in node_outputs or succ_id in ready_nodes or succ_id in node_yields:
                            continue
                        succ_input = _check_ready(succ_id, edges_by_target, node_outputs)
                        if succ_input is not None:
                            ready_nodes[succ_id] = succ_input

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


__all__ = [
    "TopologicalExecutionAlgorithm",
]
