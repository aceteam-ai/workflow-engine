# tests/test_env.py
"""Tests for environment-variable resolution on the context classes.

``ValidationContext.get_env`` is the single resolver; ``ExecutionContext.get_env``
delegates to it. Both are awaitable so that interactive contexts can raise
``ShouldYield`` to pause and request a missing variable from the user.
"""

import pytest
from overrides import override

from workflow_engine import ShouldYield
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.context import ValidationContext


class TestValidationContextGetEnv:
    @pytest.mark.asyncio
    async def test_returns_set_variable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WENGINE_TEST_VAR", "value")
        assert await ValidationContext().get_env("WENGINE_TEST_VAR") == "value"

    @pytest.mark.asyncio
    async def test_returns_default_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("WENGINE_TEST_VAR", raising=False)
        assert (
            await ValidationContext().get_env("WENGINE_TEST_VAR", "fallback")
            == "fallback"
        )

    @pytest.mark.asyncio
    async def test_raises_when_unset_without_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("WENGINE_TEST_VAR", raising=False)
        with pytest.raises(ValueError):
            await ValidationContext().get_env("WENGINE_TEST_VAR")


class TestExecutionContextDelegates:
    @pytest.mark.asyncio
    async def test_delegates_to_validation_context(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("WENGINE_TEST_VAR", "value")
        context = InMemoryExecutionContext()
        assert await context.get_env("WENGINE_TEST_VAR") == "value"

    @pytest.mark.asyncio
    async def test_yield_propagates_through_execution_context(self):
        """An interactive validation context can pause resolution via ShouldYield."""

        class InteractiveValidationContext(ValidationContext):
            @override
            async def get_env(self, key: str, default: str | None = None) -> str:
                raise ShouldYield(f"waiting for {key}")

        context = InMemoryExecutionContext(
            validation_context=InteractiveValidationContext()
        )
        with pytest.raises(ShouldYield):
            await context.get_env("NEEDS_USER_INPUT")
