# workflow_engine/contexts/local.py
import json
import os
import uuid
from collections.abc import Mapping
from typing import TypeVar

from overrides import override

from ..core import (
    Context,
    Data,
    DataMapping,
    FileValue,
    Node,
    UserException,
    Workflow,
    WorkflowErrors,
    WorkflowExecutionResult,
)
from ..core.values import dump_data_mapping, get_data_dict, serialize_data_mapping

F = TypeVar("F", bound=FileValue)


class LocalContext(Context):
    """
    A context that uses the local filesystem to store files.
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        base_dir: str = "./local",
    ):
        if run_id is None:
            run_dir: str | None = None
            while run_dir is None or os.path.exists(run_dir):
                run_id = str(uuid.uuid4())
                run_dir = os.path.join(base_dir, run_id)
        else:
            run_dir = os.path.join(base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        self.run_id = run_id
        self.run_dir = run_dir

        self.files_dir = os.path.join(self.run_dir, "files")
        self.input_dir = os.path.join(self.run_dir, "input")
        self.output_dir = os.path.join(self.run_dir, "output")
        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def _idempotent_write(self, path: str, data: str):
        if os.path.exists(path):
            with open(path, "r") as f:
                assert f.read() == data
        else:
            with open(path, "x") as f:
                f.write(data)

    def get_file_path(self, path: str) -> str:
        return os.path.join(self.files_dir, path)

    @property
    def workflow_path(self) -> str:
        return os.path.join(self.run_dir, "workflow.json")

    @property
    def workflow_input_path(self) -> str:
        return os.path.join(self.run_dir, "input.json")

    @property
    def workflow_error_path(self) -> str:
        return os.path.join(self.run_dir, "error.json")

    @property
    def workflow_output_path(self) -> str:
        return os.path.join(self.run_dir, "output.json")

    def get_node_input_path(self, node_id: str) -> str:
        return os.path.join(self.input_dir, f"{node_id}.json")

    def node_error_path(self, node_id: str) -> str:
        return os.path.join(self.run_dir, f"{node_id}.error.json")

    def get_node_output_path(self, node_id: str) -> str:
        return os.path.join(self.output_dir, f"{node_id}.json")

    @override
    async def read(
        self,
        file: FileValue,
    ) -> bytes:
        path = self.get_file_path(file.path)
        if not os.path.exists(path):
            raise UserException(f"File {file.path} not found")
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception as e:
            raise UserException(f"Failed to read file {file.path}") from e

    @override
    async def write(
        self,
        file: F,
        content: bytes,
    ) -> F:
        path = self.get_file_path(file.path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as f:
                f.write(content)
        except Exception as e:
            raise UserException(f"Failed to write file {file.path}") from e
        return file

    @override
    async def on_node_start(
        self,
        *,
        node: Node,
        input: DataMapping,
    ) -> DataMapping | None:
        self._idempotent_write(
            path=self.get_node_input_path(node.id),
            data=serialize_data_mapping(input),
        )

        output_path = self.get_node_output_path(node.id)
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                output = node.output_type.model_validate_json(f.read())
            assert isinstance(output, Data)
            return get_data_dict(output)
        return None

    @override
    async def on_node_error(
        self,
        *,
        node: Node,
        input: DataMapping,
        exception: Exception,
    ) -> Exception | DataMapping:
        self._idempotent_write(
            path=self.node_error_path(node.id),
            data=json.dumps(exception),
        )
        return exception

    @override
    async def on_node_finish(
        self,
        *,
        node: Node,
        input: DataMapping,
        output: DataMapping,
    ) -> DataMapping:
        self._idempotent_write(
            path=self.get_node_output_path(node.id),
            data=serialize_data_mapping(output),
        )
        return output

    @override
    async def on_workflow_start(
        self,
        *,
        workflow: Workflow,
        input: DataMapping,
    ) -> WorkflowExecutionResult | None:
        """
        Triggered when a workflow is started.
        If the context already knows what the node's output will be, it can
        return the output to skip node execution.
        """
        self._idempotent_write(
            path=self.workflow_input_path,
            data=serialize_data_mapping(input),
        )

        self._idempotent_write(
            path=self.workflow_path,
            data=workflow.model_dump_json(),
        )

        output_path = self.workflow_output_path
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                output = json.load(f)
            assert isinstance(output, dict)
            return WorkflowExecutionResult.success(output)

        error_path = self.workflow_error_path
        if os.path.exists(error_path):
            with open(error_path, "r") as f:
                error_and_output = json.load(f)
            assert isinstance(error_and_output, dict)
            errors = WorkflowErrors.model_validate(error_and_output["errors"])
            output = error_and_output.get("output", {})
            node_yields = error_and_output.get("node_yields", {})
            assert isinstance(output, Mapping)
            assert isinstance(node_yields, Mapping)
            return WorkflowExecutionResult.error(
                errors=errors,
                partial_output=output,
                node_yields=node_yields,
            )

        return None

    @override
    async def on_workflow_error(
        self,
        *,
        workflow: Workflow,
        input: DataMapping,
        errors: WorkflowErrors,
        partial_output: DataMapping,
        node_yields: Mapping[str, str],
    ) -> WorkflowExecutionResult:
        self._idempotent_write(
            path=self.workflow_error_path,
            data=json.dumps(
                {
                    "errors": errors.model_dump(),
                    "output": dump_data_mapping(partial_output),
                    "node_yields": node_yields,
                }
            ),
        )
        return WorkflowExecutionResult.error(
            errors=errors,
            partial_output=partial_output,
            node_yields=node_yields,
        )

    @override
    async def on_workflow_finish(
        self,
        *,
        workflow: Workflow,
        input: DataMapping,
        output: DataMapping,
    ) -> WorkflowExecutionResult:
        self._idempotent_write(
            path=self.workflow_output_path,
            data=serialize_data_mapping(output),
        )
        return WorkflowExecutionResult.success(output)


__all__ = [
    "LocalContext",
]
