"""
Thin wrapper around the real workflow_engine package.

The autoresearch agent modifies the actual engine source code — not a copy.
This module re-exports everything needed so evaluate.py and benchmarks
can import from a single place.
"""

# Core types
from workflow_engine import (
    Context,
    Data,
    DataMapping,
    DataValue,
    Edge,
    Empty,
    FloatValue,
    InputNode,
    IntegerValue,
    Node,
    NodeTypeInfo,
    OutputNode,
    Params,
    SequenceValue,
    ShouldRetry,
    ShouldYield,
    StringValue,
    Value,
    Workflow,
    WorkflowExecutionResult,
    WorkflowExecutionResultStatus,
    WorkflowValue,
)

# Contexts
from workflow_engine.contexts import InMemoryContext

# Execution algorithms
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.execution.parallel import ParallelExecutionAlgorithm

# Built-in nodes
from workflow_engine.nodes import (
    AddNode,
    ConstantIntegerNode,
    ConstantStringNode,
    ForEachNode,
    SumNode,
)

# Value helpers
from workflow_engine.core.values import get_data_dict

__all__ = [
    "AddNode",
    "ConstantIntegerNode",
    "ConstantStringNode",
    "Context",
    "Data",
    "DataMapping",
    "DataValue",
    "Edge",
    "Empty",
    "FloatValue",
    "ForEachNode",
    "get_data_dict",
    "InMemoryContext",
    "InputNode",
    "IntegerValue",
    "Node",
    "NodeTypeInfo",
    "OutputNode",
    "Params",
    "ParallelExecutionAlgorithm",
    "SequenceValue",
    "ShouldRetry",
    "ShouldYield",
    "StringValue",
    "SumNode",
    "TopologicalExecutionAlgorithm",
    "Value",
    "Workflow",
    "WorkflowExecutionResult",
    "WorkflowExecutionResultStatus",
    "WorkflowValue",
]
