# workflow_engine/execution/topological.py
"""
Topological execution algorithm with retry and rate limiting support.
"""

import asyncio

from overrides import override

from ..core import (
    DataMapping,
    ExecutionAlgorithm,
    ExecutionContext,
    ValidatedWorkflow,
    WorkflowErrorsBuilder,
    WorkflowException,
    WorkflowExecutionResult,
)
from ..core.boundary import ErrorBoundaryNode
from ..core.error import ShouldRetry, ShouldYield
from ..core.replacement import NodeReplacement
from .boundary import (
    BoundaryTracker,
    abandon_retry_if_blocked,
    flush_cancellations,
    handle_failure,
)
from .rate_limit import RateLimitRegistry
from .replacement import ReplacementGraph, ReplacementTracker, ResumeReplacement
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
        max_replacement_hops: int = 256,
    ):
        """
        Initialize the execution algorithm.

        max_retries: default maximum retry attempts for nodes (can be overridden
                     per node instance or via NodeTypeInfo.max_retries)
        rate_limits: registry of rate limit configurations per node type
        """
        if max_replacement_hops < 1:
            raise ValueError("max_replacement_hops must be positive")
        self.max_replacement_hops = max_replacement_hops
        self.max_retries = max_retries
        self.rate_limits = rate_limits or RateLimitRegistry()

    def _get_node_max_retries(self, node) -> int | None:
        """Prefer an instance budget, then the type budget; None uses the run default."""
        if node.max_retries is not None:
            return node.max_retries
        if hasattr(node, "TYPE_INFO") and node.TYPE_INFO.max_retries is not None:
            return node.TYPE_INFO.max_retries
        return None

    @override
    async def execute(
        self,
        *,
        context: ExecutionContext,
        workflow: ValidatedWorkflow,
        input: DataMapping,
    ) -> WorkflowExecutionResult:
        self.require_validated(workflow)
        async with context.execution_scope(
            legacy=self.rate_limits.configs(),
            legacy_coordinator=self.rate_limits.coordinator,
        ):
            return await self._execute(context=context, workflow=workflow, input=input)

    async def _execute(
        self,
        *,
        context: ExecutionContext,
        workflow: ValidatedWorkflow,
        input: DataMapping,
    ) -> WorkflowExecutionResult:
        result = await context.on_workflow_start(workflow=workflow, input=input)
        if result is not None:
            return result

        node_outputs: dict[str, DataMapping] = {}
        errors = WorkflowErrorsBuilder()
        retry_tracker = RetryTracker(default_max_retries=self.max_retries)
        tracker = BoundaryTracker()
        replacements = ReplacementTracker(self.max_replacement_hops)
        workflow = ReplacementGraph.model_validate(
            {key: getattr(workflow, key) for key in ValidatedWorkflow.model_fields}
        )

        # Track nodes that are waiting for retry (node_id -> input)
        pending_retry: dict[str, DataMapping] = {}
        # Track nodes that yielded (node_id -> yield message)
        node_yields: dict[str, str] = {}

        try:
            try:
                ready_nodes = dict(workflow.get_initial_ready_nodes(input))

                while len(ready_nodes) > 0 or len(pending_retry) > 0:
                    # Check if any pending retries are now ready
                    for node_id in list(pending_retry.keys()):
                        state = retry_tracker.get_state(node_id)
                        if state.is_ready():
                            ready_nodes[node_id] = pending_retry.pop(node_id)

                    # If no nodes are ready, wait for the shortest backoff
                    if len(ready_nodes) == 0 and len(pending_retry) > 0:
                        wait_time = retry_tracker.min_wait_time()
                        if wait_time and wait_time.total_seconds() > 0:
                            await asyncio.sleep(wait_time.total_seconds())
                        continue

                    if len(ready_nodes) == 0:
                        break

                    node_id, node_input = ready_nodes.popitem()
                    if not retry_tracker.get_state(node_id).is_ready():
                        pending_retry[node_id] = node_input
                        continue
                    node = workflow.nodes_by_id[node_id]
                    input_type = workflow.node_input_types[node_id]
                    output_type = workflow.node_output_types[node_id]

                    expanded = False
                    replaced = False
                    completed = {node_id}
                    failure: WorkflowException | None = None
                    try:
                        if node_id in replacements.completion_sources:
                            node_result = ResumeReplacement()
                        else:
                            node_result = await node(
                                context=context,
                                input_type=input_type,
                                output_type=output_type,
                                input=node_input,
                                allow_replacement=True,
                            )

                        if isinstance(node_result, ResumeReplacement):
                            completed, adaptation_errors = await replacements.complete(
                                [replacements.completion_sources[node_id]],
                                node_outputs,
                                context,
                                tracker,
                                retry_tracker=retry_tracker,
                                pending_retry=pending_retry,
                                node_yields=node_yields,
                            )
                            if adaptation_errors:
                                failure = adaptation_errors[0]
                        elif isinstance(node_result, NodeReplacement):
                            (
                                workflow,
                                replacement_ready,
                                recovered,
                            ) = await replacements.install(
                                workflow,
                                node,
                                node_result,
                                context,
                                retry_tracker,
                                tracker,
                            )
                            ready_nodes.update(replacement_ready)
                            replaced = recovered is None
                            if replaced:
                                await asyncio.sleep(0)
                            if recovered is not None:
                                node_outputs[node_id] = recovered
                                (
                                    completed,
                                    adaptation_errors,
                                ) = await replacements.complete(
                                    [node_id],
                                    node_outputs,
                                    context,
                                    tracker,
                                    retry_tracker=retry_tracker,
                                    pending_retry=pending_retry,
                                    node_yields=node_yields,
                                )
                                if adaptation_errors:
                                    failure = adaptation_errors[0]
                        elif isinstance(node_result, ValidatedWorkflow):
                            if isinstance(node, ErrorBoundaryNode):
                                tracker.register(
                                    node_id=node_id,
                                    node=node,
                                    input=node_input,
                                    input_type=input_type,
                                    output_type=output_type,
                                    subgraph=node_result,
                                )
                            workflow = workflow.expand_node(node_id, node_result)
                            replacements.expanded(node_id, node_result)
                            expanded = True
                        else:
                            node_outputs[node.id] = node_result
                            completed, adaptation_errors = await replacements.complete(
                                [node_id],
                                node_outputs,
                                context,
                                tracker,
                                retry_tracker=retry_tracker,
                                pending_retry=pending_retry,
                                node_yields=node_yields,
                            )
                            if adaptation_errors:
                                failure = adaptation_errors[0]

                    except ShouldYield as e:
                        node_yields[node_id] = e.message
                        await context.on_node_yield(
                            node=node,
                            input_type=input_type,
                            output_type=output_type,
                            input=node_input,
                            exception=e,
                        )
                        continue

                    except ShouldRetry as e:
                        node_max_retries = self._get_node_max_retries(node)

                        if await abandon_retry_if_blocked(
                            tracker, workflow, context, node_id, node_input
                        ):
                            continue

                        if retry_tracker.should_retry(node_id, node_max_retries):
                            retry_tracker.record_retry(node_id, e)
                            await replacements.record_retry(
                                node_id, retry_tracker, context
                            )
                            pending_retry[node_id] = node_input

                            # Call the on_node_retry hook
                            state = retry_tracker.get_state(node_id)
                            await context.on_node_retry(
                                node=node,
                                input_type=input_type,
                                output_type=output_type,
                                input=node_input,
                                exception=e,
                                attempt=state.attempt,
                            )

                            # Continue to next node without re-raising
                            continue

                        # Max retries exceeded: contain in a boundary if any,
                        # otherwise surface as a workflow error.
                        failure = e

                    except WorkflowException as e:
                        failure = e

                    if failure is not None:
                        contained = await handle_failure(
                            tracker,
                            workflow,
                            context,
                            failure.node_id or node_id,
                            failure,
                            ready_nodes=ready_nodes,
                            pending_retry=pending_retry,
                            retry_tracker=retry_tracker,
                        )
                        await replacements.failed(failure, tracker, context)
                        if not contained:
                            raise failure
                    elif expanded:
                        ready_nodes = {
                            node_id: node_input
                            for node_id, node_input in workflow.get_ready_nodes(
                                node_outputs=node_outputs,
                                partial_results=ready_nodes,
                            ).items()
                            if node_id not in node_yields
                            and not tracker.is_blocked(node_id)
                        }
                    elif not replaced:
                        ready_nodes.update(
                            {
                                nid: inp
                                for nid, inp in workflow.get_ready_successors(
                                    completed,
                                    node_outputs,
                                    skip=set(node_outputs)
                                    | set(ready_nodes)
                                    | set(node_yields),
                                ).items()
                                if not tracker.is_blocked(nid)
                            }
                        )

                    (
                        completed_boundaries,
                        adaptation_errors,
                    ) = await replacements.materialize_pending(
                        workflow,
                        context,
                        tracker,
                        node_outputs,
                        in_flight=frozenset(),
                        node_yields=node_yields,
                        ready_nodes=ready_nodes,
                        pending_retry=pending_retry,
                        retry_tracker=retry_tracker,
                    )
                    if adaptation_errors:
                        raise adaptation_errors[0]
                    ready_nodes.update(
                        {
                            nid: inp
                            for nid, inp in workflow.get_ready_successors(
                                completed_boundaries,
                                node_outputs,
                                skip=set(node_outputs)
                                | set(ready_nodes)
                                | set(node_yields),
                            ).items()
                            if not tracker.is_blocked(nid)
                        }
                    )

                # Held boundaries (a member yielded) report the members that
                # will not run this pass, even though they have not
                # materialized. Nothing new can become ready after this
                # point in the pass; node_yields is non-empty whenever a
                # pending boundary remains (yield is the only thing that can
                # keep can_materialize() false forever in this executor,
                # since in_flight is always empty here).
                for b in tracker.pending():
                    assert len(node_yields) > 0
                    await flush_cancellations(
                        tracker,
                        workflow,
                        context,
                        b,
                        node_outputs=node_outputs,
                        in_flight=frozenset(),
                        node_yields=frozenset(node_yields),
                    )

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
                # other errors pass through as WorkflowExceptions
                if isinstance(e, WorkflowException):
                    raise
                else:
                    raise WorkflowException.for_operator(
                        f"Unhandled exception in workflow: {e}",
                    ) from e
        except WorkflowException as e:
            errors.add(e)
            partial_output = await workflow.get_output(
                context=context,
                node_outputs=node_outputs,
                partial=True,
            )
            result = await context.on_workflow_error(
                workflow=workflow,
                input=input,
                errors=errors.build(),
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
