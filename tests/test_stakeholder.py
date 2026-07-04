import pytest

from workflow_engine.core.stakeholder import StakeholderLevel

pytestmark = pytest.mark.unit

# Semantic order from most to least powerful.
_ORDERED = (
    StakeholderLevel.ENGINEER,
    StakeholderLevel.OPERATOR,
    StakeholderLevel.BUILDER,
    StakeholderLevel.USER,
)


def test_strict_ordering_matches_semantic_order():
    for i, lower in enumerate(_ORDERED):
        for higher in _ORDERED[i + 1 :]:
            assert lower < higher
            assert higher > lower
            assert not (higher < lower)
            assert not (lower > higher)


def test_non_strict_ordering_agrees_with_strict():
    for i, lower in enumerate(_ORDERED):
        for higher in _ORDERED[i + 1 :]:
            assert lower <= higher
            assert higher >= lower
            assert not (higher <= lower)
            assert not (lower >= higher)


def test_non_strict_ordering_reflexive():
    for level in _ORDERED:
        assert level <= level
        assert level >= level
        assert not (level < level)
        assert not (level > level)
