"""Tests for the engine.yaml schema (WorkflowEngineConfig, NodesConfig)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from workflow_engine.core.config import (
    ENGINE_YAML_NAME,
    EntryPointRef,
    NodesConfig,
    WorkflowEngineConfig,
)
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.nodes.arithmetic import AddNode, SumNode
from workflow_engine.nodes.iteration import ForEachNode


class TestEntryPointRef:
    def test_splits_distribution_and_entry_point(self):
        ref = EntryPointRef(root="aceteam-workflow-engine:Sum")
        assert ref.distribution == "aceteam-workflow-engine"
        assert ref.entry_point_name == "Sum"

    def test_resolves_to_node_class(self):
        ref = EntryPointRef(root="aceteam-workflow-engine:Sum")
        assert ref.node_cls is SumNode

    def test_normalizes_distribution_name(self):
        # PEP 503: underscores/dots/hyphens collapse, case-insensitive.
        ref = EntryPointRef(root="Aceteam_Workflow.Engine:ForEach")
        assert ref.node_cls is ForEachNode

    def test_rejects_malformed_string(self):
        with pytest.raises(ValidationError):
            EntryPointRef(root="no-colon-here")

    def test_unknown_entry_point_raises_lookup_error(self):
        ref = EntryPointRef(root="aceteam-workflow-engine:DoesNotExist")
        with pytest.raises(LookupError, match="DoesNotExist"):
            _ = ref.node_cls


class TestNodesConfigGrammar:
    def test_accepts_global_glob_string(self):
        nc = NodesConfig.model_validate({"*": "aceteam-workflow-engine"})
        assert nc.global_glob_distributions == ("aceteam-workflow-engine",)

    def test_accepts_global_glob_list(self):
        nc = NodesConfig.model_validate(
            {"*": ["aceteam-workflow-engine", "other-dist"]}
        )
        assert nc.global_glob_distributions == (
            "aceteam-workflow-engine",
            "other-dist",
        )

    def test_accepts_prefix_glob(self):
        nc = NodesConfig.model_validate({"vendor/legacy:*": "aceteam-workflow-engine"})
        assert nc.prefix_glob_distributions == {
            "vendor/legacy": ("aceteam-workflow-engine",)
        }

    def test_accepts_explicit_bare(self):
        nc = NodesConfig.model_validate({"Sum": "aceteam-workflow-engine:Sum"})
        assert "Sum" in nc.explicit_entries

    def test_accepts_explicit_prefixed(self):
        nc = NodesConfig.model_validate(
            {"vendor/legacy:Sum": "aceteam-workflow-engine:Sum"}
        )
        assert "vendor/legacy:Sum" in nc.explicit_entries

    def test_rejects_invalid_key(self):
        with pytest.raises(ValidationError):
            NodesConfig.model_validate({"!bad!": "aceteam-workflow-engine:Sum"})

    def test_rejects_non_string_explicit_value(self):
        with pytest.raises(ValidationError):
            NodesConfig.model_validate({"Sum": ["a", "b"]})

    def test_rejects_explicit_value_without_colon(self):
        with pytest.raises(ValidationError):
            NodesConfig.model_validate({"Sum": "aceteam-workflow-engine"})


class TestNodesConfigResolution:
    def test_global_glob_mounts_every_node_bare(self):
        nc = NodesConfig.model_validate({"*": "aceteam-workflow-engine"})
        reg = nc.node_registry
        # Spot-check a few representative builtins resolve to the right classes.
        assert reg.get("Sum") is SumNode
        assert reg.get("Add") is AddNode
        assert reg.get("ForEach") is ForEachNode
        assert reg.get("Input") is InputNode
        assert reg.get("Output") is OutputNode

    def test_explicit_overrides_glob(self):
        # Map "Sum" explicitly to Add's entry point — explicit wins over the
        # glob's natural "Sum" mounting.
        nc = NodesConfig.model_validate(
            {
                "*": "aceteam-workflow-engine",
                "Sum": "aceteam-workflow-engine:Add",
            }
        )
        assert nc.node_registry.get("Sum") is AddNode

    def test_prefix_glob_keyspace_does_not_collide_with_bare(self):
        nc = NodesConfig.model_validate(
            {
                "*": "aceteam-workflow-engine",
                "vendor/legacy:*": "aceteam-workflow-engine",
            }
        )
        reg = nc.node_registry
        assert reg.get("Sum") is SumNode
        assert reg.get("vendor/legacy:Sum") is SumNode

    def test_unmapped_name_is_not_in_registry(self):
        nc = NodesConfig.model_validate({"Sum": "aceteam-workflow-engine:Sum"})
        reg = nc.node_registry
        assert reg.get("Sum") is SumNode
        assert reg.get("Add") is None


class TestWorkflowEngineConfig:
    def test_load_yaml(self, tmp_path: Path):
        path = tmp_path / ENGINE_YAML_NAME
        path.write_text('schema_version: 1\nnodes:\n  "*": aceteam-workflow-engine\n')
        config = WorkflowEngineConfig.load(path)
        assert config.schema_version == 1
        assert config.node_registry.get("Sum") is SumNode

    def test_load_rejects_wrong_schema_version(self, tmp_path: Path):
        path = tmp_path / ENGINE_YAML_NAME
        path.write_text('schema_version: 999\nnodes:\n  "*": aceteam-workflow-engine\n')
        with pytest.raises(ValidationError):
            WorkflowEngineConfig.load(path)

    def test_schema_version_defaults_to_1(self):
        config = WorkflowEngineConfig.model_validate(
            {"nodes": {"*": "aceteam-workflow-engine"}}
        )
        assert config.schema_version == 1


class TestFindEngineYaml:
    def test_finds_in_current_directory(self, tmp_path: Path):
        (tmp_path / ENGINE_YAML_NAME).write_text("schema_version: 1\nnodes: {}\n")
        found = WorkflowEngineConfig.find_engine_yaml(start=tmp_path)
        assert found == (tmp_path / ENGINE_YAML_NAME).resolve()

    def test_walks_up_to_ancestor(self, tmp_path: Path):
        (tmp_path / ENGINE_YAML_NAME).write_text("schema_version: 1\nnodes: {}\n")
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        found = WorkflowEngineConfig.find_engine_yaml(start=deep)
        assert found == (tmp_path / ENGINE_YAML_NAME).resolve()

    def test_returns_none_when_absent(self, tmp_path: Path):
        # tmp_path is somewhere under /tmp; no engine.yaml expected on the way up.
        found = WorkflowEngineConfig.find_engine_yaml(start=tmp_path)
        assert found is None

    def test_accepts_file_as_start(self, tmp_path: Path):
        (tmp_path / ENGINE_YAML_NAME).write_text("schema_version: 1\nnodes: {}\n")
        marker_file = tmp_path / "some_file.txt"
        marker_file.write_text("")
        found = WorkflowEngineConfig.find_engine_yaml(start=marker_file)
        assert found == (tmp_path / ENGINE_YAML_NAME).resolve()
