from enum import StrEnum


class StakeholderLevel(StrEnum):
    """The level of a stakeholder.

    The lower the level, the more "powerful" the stakeholder is when it comes to
    fixing errors and changing the behaviour of workflows.
    """

    # Developers of workflow-engine itself (engineers).
    ENGINEER = "ENGINEER"

    # Uses workflow-engine to define their own node types, value types, contexts, and execution algorithms.
    # May expose their engine as a library to other operators.
    # May provide builders with interfaces for workflow construction.
    OPERATOR = "OPERATOR"

    # Builds workflows to be run on the operator's engine.
    # May share workflows with other builders.
    # May provide users with interfaces to run their workflows.
    BUILDER = "BUILDER"

    # Runs workflows built by builders.
    USER = "USER"

    def __lt__(self, other: "StakeholderLevel") -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _STAKEHOLDER_LEVELS.index(self) < _STAKEHOLDER_LEVELS.index(other)

    def __gt__(self, other: "StakeholderLevel") -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _STAKEHOLDER_LEVELS.index(self) > _STAKEHOLDER_LEVELS.index(other)


_STAKEHOLDER_LEVELS = (
    StakeholderLevel.ENGINEER,
    StakeholderLevel.OPERATOR,
    StakeholderLevel.BUILDER,
    StakeholderLevel.USER,
)

__all__ = [
    "StakeholderLevel",
]
