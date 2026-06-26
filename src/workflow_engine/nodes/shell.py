# workflow_engine/nodes/shell.py
"""
A node for running arbitrary shell commands.

Every shell command can be viewed as a workflow node: a rendered command line
runs in a subprocess, and its standard streams become output files. Arguments
are passed into the command via Jinja substitution from the node's input.

.. warning::
    This node executes arbitrary shell commands in the same environment as the
    engine itself. It is effectively remote code execution and must only be
    enabled for trusted, locally-operated engines. Never expose this node on an
    engine that runs untrusted workflows or accepts workflows over the network.
"""

import asyncio
import os
from typing import ClassVar, Type

from jinja2 import StrictUndefined, Template
from overrides import override
from pydantic import Field

from ..core import (
    BooleanValue,
    Data,
    ExecutionContext,
    File,
    IntegerValue,
    Node,
    NodeTypeInfo,
    Params,
    StringMapValue,
    StringValue,
    ValidationContext,
)
from ..files import TextFileValue


class BashInput(Data):
    arguments: StringMapValue[StringValue] = Field(
        title="Arguments",
        description=(
            "The named values substituted into the command via Jinja "
            "(e.g. `{{ name }}`). Leave empty for commands without arguments."
        ),
        default=StringMapValue({}),
    )


class BashOutput(Data):
    stdout: TextFileValue = Field(
        title="Standard Output",
        description="The text written to standard output by the command.",
    )
    stderr: TextFileValue = Field(
        title="Standard Error",
        description="The text written to standard error by the command.",
    )
    exit_code: IntegerValue = Field(
        title="Exit Code",
        description="The exit status returned by the command (0 means success).",
    )


class BashCombinedOutput(Data):
    output: TextFileValue = Field(
        title="Output",
        description=(
            "The combined text written to standard output and standard error "
            "by the command."
        ),
    )
    exit_code: IntegerValue = Field(
        title="Exit Code",
        description="The exit status returned by the command (0 means success).",
    )


class BashNodeParams(Params):
    command: StringValue = Field(
        title="Command",
        description=(
            "The shell command to run. Supports Jinja substitution from the "
            "node's arguments (e.g. `echo Hello {{ name }}`)."
        ),
    )
    combine_output: BooleanValue = Field(
        title="Combine Output",
        description=(
            "Whether to merge standard error into standard output, producing a "
            "single combined output file instead of separate ones."
        ),
        default=BooleanValue(False),
    )


class BashNode(Node[BashInput, Data, BashNodeParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Bash",
        description=(
            "Runs a shell command and captures its output. WARNING: this "
            "executes arbitrary commands in the engine's environment and must "
            "only be enabled for trusted, locally-operated engines."
        ),
        version="0.1.0",
        parameter_type=BashNodeParams,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[BashInput]:
        return BashInput

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> Type[Data]:
        if self.params.combine_output.root:
            return BashCombinedOutput
        return BashOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[BashInput],
        output_type: Type[Data],
        input: BashInput,
    ) -> Data:
        arguments = {key: value.root for key, value in input.arguments.items()}
        # NOTE: Template annotation is needed because Template.__new__ is typed
        # as returning Any to avoid breaking Sphinx.
        template: Template = Template(
            self.params.command.root,
            undefined=StrictUndefined,
        )
        command = template.render(**arguments)

        shell = os.environ.get("SHELL") or "/bin/bash"
        combine = self.params.combine_output.root
        process = await asyncio.create_subprocess_exec(
            shell,
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=(asyncio.subprocess.STDOUT if combine else asyncio.subprocess.PIPE),
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        assert process.returncode is not None, (
            "process.returncode was None after communicate()"
        )

        exit_code = IntegerValue(process.returncode)

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        if combine:
            output_file = TextFileValue(File(path=f"{self.id}.output.txt"))
            output_file = await output_file.write_text(context, text=stdout_text)
            return BashCombinedOutput(
                output=output_file,
                exit_code=exit_code,
            )

        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        stdout_file = TextFileValue(File(path=f"{self.id}.stdout.txt"))
        stdout_file = await stdout_file.write_text(context, text=stdout_text)
        stderr_file = TextFileValue(File(path=f"{self.id}.stderr.txt"))
        stderr_file = await stderr_file.write_text(context, text=stderr_text)
        return BashOutput(
            stdout=stdout_file,
            stderr=stderr_file,
            exit_code=exit_code,
        )


__all__ = [
    "BashNode",
]
