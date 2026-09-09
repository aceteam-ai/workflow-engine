"""Tests for scripts/generate_typecast_graph.py.

The graph builder is a standalone script (not part of the installed
package), so we import it directly by path rather than through the
`workflow_engine` package, matching test_no_ai_attribution_guard.py.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from workflow_engine.core.values.json import JSONValue
from workflow_engine.core.values.primitives import StringValue
from workflow_engine.core.values.result import Result
from workflow_engine.core.values.value import Value

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "generate_typecast_graph.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_typecast_graph", _SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generator = _load_module()


# `register=False` keeps these synthetic types out of the real, global
# ValueRegistry: build_typecast_graph() takes an explicit `all_classes`
# mapping, so the test can construct its own tiny universe of types instead
# of depending on (or polluting) the app's actual registered Value classes.
class _TargetValue(Value[str], register=False):
    pass


class _CastsToTarget(Value[str], register=False):
    pass


@_CastsToTarget.register_cast_to(_TargetValue)
def _cast_to_target(value: _CastsToTarget, context) -> _TargetValue:
    return _TargetValue(value.root)


class _DeclinesTarget(Value[str], register=False):
    """
    Registers a caster to _TargetValue purely to shadow some blanket caster
    it would otherwise inherit, then always declines by returning None, the
    same pattern Result uses to shadow the blanket casts to StringValue and
    JSONValue (see result.py, #232).
    """

    pass


@_DeclinesTarget.register_generic_cast_to(_TargetValue)
def _decline_cast_to_target(source_type, target_type):
    return None


def test_caster_that_returns_a_cast_produces_an_edge():
    all_classes = {
        "_TargetValue": _TargetValue,
        "_CastsToTarget": _CastsToTarget,
    }

    graph = generator.build_typecast_graph(all_classes)

    assert graph.has_edge("_CastsToTarget", "_TargetValue")


def test_caster_that_always_declines_produces_no_edge():
    """
    Pins the #239 fix: a registered caster *key* is not enough to draw an
    edge. _DeclinesTarget registers a caster to _TargetValue (so the key is
    present, exactly like the pre-fix bug would have keyed off), but the
    caster always returns None, so can_cast_to() is False and no edge should
    be drawn, matching Result's real shadowing casters to StringValue and
    JSONValue.
    """
    all_classes = {
        "_TargetValue": _TargetValue,
        "_DeclinesTarget": _DeclinesTarget,
    }

    graph = generator.build_typecast_graph(all_classes)

    # The node itself is still present...
    assert graph.has_node("_DeclinesTarget")
    # ...but the declining caster produced no outgoing edge.
    assert not graph.has_edge("_DeclinesTarget", "_TargetValue")
    assert list(graph.out_edges("_DeclinesTarget")) == []


def test_caster_key_pointing_outside_all_classes_produces_no_edge():
    """
    A registered caster whose target isn't itself a node in `all_classes`
    (e.g. filtered out elsewhere, such as an unbound generic) must not crash
    or add a dangling edge.
    """
    all_classes = {
        "_CastsToTarget": _CastsToTarget,
    }

    graph = generator.build_typecast_graph(all_classes)

    assert graph.has_node("_CastsToTarget")
    assert list(graph.out_edges("_CastsToTarget")) == []


def test_result_declines_casting_to_string_or_json():
    """
    The exact scenario from #239, against production code: Result registers
    casters to StringValue and JSONValue purely to shadow the blanket "cast
    anything" casters it would otherwise inherit (see result.py, #232), and
    those casters always return None. can_cast_to() is exactly what
    build_typecast_graph() consults per candidate edge (see
    test_caster_that_always_declines_produces_no_edge above for the
    synthetic version of this same mechanism), so this pins the concrete
    production values the fix targets.

    Note: Result itself is never a *node* in the real graph, since it's an
    unbound generic (Value.__init_subclass__ only registers fully-concrete
    classes) -- so this asserts the underlying can_cast_to() behavior
    directly rather than routing bare Result through build_typecast_graph(),
    which would also need it as a target for its own Result -> Result
    caster, and that caster assumes a parameterized Result[T].
    """
    assert not Result.can_cast_to(StringValue)
    assert not Result.can_cast_to(JSONValue)
