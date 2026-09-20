"""Integration tests for pool lifecycle using mocked asyncpg pools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import DatabaseConfig
from pg_mcp.db.pool import close_pools, create_pools


def _config(name: str) -> DatabaseConfig:
    return DatabaseConfig(name=name, host="localhost", user="postgres", password="postgres")


@pytest.mark.asyncio
async def test_create_pools_maps_by_database_name() -> None:
    """create_pools returns one pool per unique database name."""
    configs = [_config("blog_small"), _config("ecommerce_medium")]

    async def fake_create_pool(config: DatabaseConfig) -> MagicMock:
        pool = MagicMock()
        pool.db_name = config.name
        pool.close = AsyncMock()
        return pool

    with patch("pg_mcp.db.pool.create_pool", side_effect=fake_create_pool):
        pools = await create_pools(configs)

    assert set(pools) == {"blog_small", "ecommerce_medium"}
    assert pools["blog_small"].db_name == "blog_small"
    assert pools["ecommerce_medium"].db_name == "ecommerce_medium"


@pytest.mark.asyncio
async def test_create_pools_closes_successful_pools_on_failure() -> None:
    """If one pool fails, already-created pools are closed."""
    configs = [_config("ok_db"), _config("bad_db")]
    closed: list[str] = []

    async def fake_create_pool(config: DatabaseConfig) -> MagicMock:
        if config.name == "bad_db":
            raise RuntimeError("connection refused")
        pool = MagicMock()
        pool.db_name = config.name

        async def _close() -> None:
            closed.append(config.name)

        pool.close = _close
        return pool

    with (
        patch("pg_mcp.db.pool.create_pool", side_effect=fake_create_pool),
        pytest.raises(RuntimeError, match="connection refused"),
    ):
        await create_pools(configs)

    assert closed == ["ok_db"]


@pytest.mark.asyncio
async def test_close_pools_calls_close() -> None:
    """close_pools should close every pool."""
    pool = MagicMock()
    pool.close = AsyncMock()
    await close_pools({"blog_small": pool}, timeout=1.0)
    pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_pool_disables_ssl_for_localhost() -> None:
    """Local connections must not use sslmode=prefer (Windows SSL false positives)."""
    from pg_mcp.db.pool import create_pool

    pool = MagicMock()
    with patch("pg_mcp.db.pool.asyncpg.create_pool", new_callable=AsyncMock) as mocked:
        mocked.return_value = pool
        result = await create_pool(_config("blog_small"))

    assert result is pool
    mocked.assert_awaited_once()
    assert mocked.await_args.kwargs["ssl"] is False
