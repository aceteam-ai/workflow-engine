# tests/test_node_versioning.py
import warnings

import pytest
from pydantic import ValidationError

from workflow_engine import StringValue
from workflow_engine.nodes import ConstantStringNode
from workflow_engine.nodes.constant import ConstantString
from workflow_engine.utils.semver import LATEST_SEMANTIC_VERSION


def _make_node(**kwargs) -> ConstantStringNode:
    return ConstantStringNode(
        id="test_node",
        type="ConstantString",
        params=ConstantString(value=StringValue("test")),
        **kwargs,
    )


class TestNodeVersioning:
    """Test node versioning functionality using ConstantStringNode."""

    @pytest.mark.unit
    def test_default_version_when_none_provided(self):
        """Test that when no version is provided, the node defaults to the current version."""
        node = _make_node()

        assert node.version == "0.4.0"
        assert node.version_tuple == (0, 4, 0)

    @pytest.mark.unit
    def test_serialization_includes_version(self):
        """Test that serializing the node includes the version."""
        node = _make_node()

        serialized = node.model_dump()

        assert "version" in serialized
        assert serialized["version"] == "0.4.0"

        json_str = node.model_dump_json()
        assert '"version":"0.4.0"' in json_str

    @pytest.mark.unit
    def test_older_version_triggers_warning(self):
        """Test that providing an older version triggers a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            node = _make_node(version="0.3.14")

            assert len(w) == 1
            warning = w[0]
            assert issubclass(warning.category, UserWarning)

            warning_message = str(warning.message)
            assert (
                "Node version 0.3.14 is older than the latest version (0.4.0) supported by this workflow engine instance, and may need to be migrated."
                in warning_message
            )

            assert node.version == "0.3.14"
            assert node.version_tuple == (0, 3, 14)

    @pytest.mark.unit
    def test_newer_version_throws_error(self):
        """Test that providing a newer version throws an error."""
        with pytest.raises(ValidationError) as exc_info:
            _make_node(version="0.5.0")

        error_message = str(exc_info.value)
        assert (
            "Node version 0.5.0 is newer than the latest version (0.4.0) supported by this workflow engine instance."
            in error_message
        )

    @pytest.mark.unit
    def test_same_version_no_warning(self):
        """Test that providing the same version as current doesn't trigger warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            node = _make_node(version="0.4.0")

            assert len(w) == 0
            assert node.version == "0.4.0"

    @pytest.mark.unit
    def test_latest_version_constant(self):
        """Test that using LATEST_SEMANTIC_VERSION constant works correctly."""
        node = _make_node(version=LATEST_SEMANTIC_VERSION)

        assert node.version == "0.4.0"
        assert node.version_tuple == (0, 4, 0)

    @pytest.mark.unit
    def test_version_validation_during_deserialization(self):
        """Test that version validation works during deserialization."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            node_data = {
                "id": "test_node",
                "type": "ConstantString",
                "version": "0.3.0",
                "params": {"value": "test"},
            }

            node = ConstantStringNode.model_validate(node_data)
            assert len(w) == 1
            assert node.version == "0.3.0"

        with pytest.raises(ValidationError):
            node_data = {
                "id": "test_node",
                "type": "ConstantString",
                "version": "0.5.0",
                "params": {"value": "test"},
            }
            ConstantStringNode.model_validate(node_data)
