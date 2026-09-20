"""Unit tests for tracing context and metrics helpers."""

import pytest

from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import generate_request_id, get_request_id, request_context
from pg_mcp.server import health


class TestTracing:
    """Request ID should propagate through the async context."""

    @pytest.mark.asyncio
    async def test_request_context_sets_and_resets_id(self) -> None:
        """request_context yields an id and clears it on exit."""
        assert get_request_id() is None
        async with request_context() as request_id:
            assert request_id
            assert get_request_id() == request_id
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_request_context_accepts_explicit_id(self) -> None:
        """An explicit request id is preserved."""
        explicit = generate_request_id()
        async with request_context(explicit) as request_id:
            assert request_id == explicit
            assert get_request_id() == explicit


class TestMetricsCollector:
    """Metrics helper methods should be callable."""

    def test_increment_helpers(self) -> None:
        """Counter helpers do not raise."""
        collector = MetricsCollector()
        collector.increment_query_request("success", "blog_small")
        collector.increment_llm_call("generate_sql")
        collector.observe_llm_latency("generate_sql", 0.01)
        collector.increment_sql_rejected("blocked_table")
        collector.observe_db_query_duration(0.02)


class TestHealthTool:
    """Health tool reports uninitialized state outside lifespan."""

    @pytest.mark.asyncio
    async def test_health_not_initialized(self) -> None:
        """Without lifespan startup the tool reports not_initialized."""
        result = await health()
        assert result["status"] == "not_initialized"
        assert result["databases"] == []
