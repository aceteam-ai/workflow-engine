"""Tests for the engine.yaml schema (WorkflowEngineConfig, NodesConfig)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from workflow_engine.core.config import (
    Distribution,
    EntryPointRef,
    NodesConfig,
    WorkflowEngineConfig,
)
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.nodes.arithmetic import AddNode, SumNode
from workflow_engine.nodes.iteration import ForEachNode


class TestDistribution:
    def test_equality_is_canonical(self):
        assert Distribution("Acme.Scrapers") == Distribution("acme-scrapers")

    def test_hashes_alike_in_sets(self):
        assert len({Distribution("Acme.Scrapers"), Distribution("acme-scrapers")}) == 1

    def test_str_keeps_original_spelling(self):
        assert str(Distribution("Acme.Scrapers")) == "Acme.Scrapers"

    def test_not_equal_to_bare_string(self):
        assert Distribution("acme") != "acme"

    def test_from_requirement_strips_extras_and_version(self):
        assert Distribution.from_requirement("acme-scrapers[x]>=1.4") == Distribution(
            "acme-scrapers"
        )


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

    def test_rejects_invalid_distribution_name_in_glob(self):
        with pytest.raises(ValidationError):
            NodesConfig.model_validate({"*": "not a valid dist name!"})

    def test_rejects_invalid_distribution_name_in_glob_list(self):
        with pytest.raises(ValidationError):
            NodesConfig.model_validate({"*": ["aceteam-workflow-engine", "bad name!"]})


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

    def test_same_distribution_different_spellings_is_not_collision(self):
        # PEP 503: "Aceteam_Workflow.Engine" normalizes to the same name as
        # "aceteam-workflow-engine". Listing both in `"*"` must not flag every
        # node as a bare-name collision.
        nc = NodesConfig.model_validate(
            {"*": ["aceteam-workflow-engine", "Aceteam_Workflow.Engine"]}
        )
        assert nc.node_registry.get("Sum") is SumNode

    def test_unmapped_name_is_not_in_registry(self):
        nc = NodesConfig.model_validate({"Sum": "aceteam-workflow-engine:Sum"})
        reg = nc.node_registry
        assert reg.get("Sum") is SumNode
        assert reg.get("Add") is None


class TestWorkflowEngineConfig:
    def test_load_yaml(self, tmp_path: Path):
        path = tmp_path / "engine.yaml"
        path.write_text('schema_version: 1\nnodes:\n  "*": aceteam-workflow-engine\n')
        config = WorkflowEngineConfig.load(path)
        assert config.schema_version == 1
        assert config.node_registry.get("Sum") is SumNode

    def test_load_rejects_wrong_schema_version(self, tmp_path: Path):
        path = tmp_path / "engine.yaml"
        path.write_text('schema_version: 999\nnodes:\n  "*": aceteam-workflow-engine\n')
        with pytest.raises(ValidationError):
            WorkflowEngineConfig.load(path)

    def test_schema_version_defaults_to_1(self):
        config = WorkflowEngineConfig.model_validate(
            {"nodes": {"*": "aceteam-workflow-engine"}}
        )
        assert config.schema_version == 1
