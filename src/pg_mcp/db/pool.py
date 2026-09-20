"""Database connection pool management.

This module provides utilities for creating and managing asyncpg connection
pools for PostgreSQL databases.
"""

import asyncio
import logging

import asyncpg
from asyncpg import Pool

from pg_mcp.config.settings import DatabaseConfig

logger = logging.getLogger(__name__)


async def create_pool(config: DatabaseConfig) -> Pool:
    """Create a connection pool for a single database.

    Args:
        config: Database configuration containing connection parameters
            and pool settings.

    Returns:
        Pool: An asyncpg connection pool instance.

    Raises:
        ConnectionError: If the database is unreachable or authentication fails.

    Example:
        >>> config = DatabaseConfig(host="localhost", name="mydb")
        >>> pool = await create_pool(config)
        >>> async with pool.acquire() as conn:
        ...     result = await conn.fetch("SELECT 1")
    """
    try:
        pool = await asyncpg.create_pool(
            host=config.host,
            port=config.port,
            database=config.name,
            user=config.user,
            password=config.password,
            min_size=config.min_pool_size,
            max_size=config.max_pool_size,
            timeout=config.pool_timeout,
            command_timeout=config.command_timeout,
            ssl=config.use_ssl,
        )
    except (OSError, TimeoutError, asyncpg.PostgresError) as exc:
        raise ConnectionError(
            f"Failed to connect to PostgreSQL at {config.safe_dsn}. "
            "Is the server running and accepting connections?"
        ) from exc

    if pool is None:
        raise RuntimeError(f"Failed to create connection pool for {config.name}")

    return pool


async def create_pools(configs: list[DatabaseConfig]) -> dict[str, Pool]:
    """Create connection pools for multiple databases.

    This function creates pools concurrently for all provided database
    configurations.

    Args:
        configs: List of database configurations.

    Returns:
        dict[str, Pool]: Dictionary mapping database names to their pools.

    Raises:
        asyncpg.PostgresError: If any database connection fails.

    Example:
        >>> configs = [
        ...     DatabaseConfig(name="db1", host="localhost"),
        ...     DatabaseConfig(name="db2", host="localhost"),
        ... ]
        >>> pools = await create_pools(configs)
        >>> assert "db1" in pools and "db2" in pools
    """
    if not configs:
        return {}

    results = await asyncio.gather(
        *[create_pool(config) for config in configs],
        return_exceptions=True,
    )

    pools: dict[str, Pool] = {}
    errors: list[BaseException] = []
    for config, result in zip(configs, results, strict=True):
        if isinstance(result, BaseException):
            errors.append(result)
        else:
            pools[config.name] = result

    if errors:
        await close_pools(pools)
        raise errors[0]

    return pools


async def close_pools(pools: dict[str, Pool], timeout: float = 10.0) -> None:  # noqa: ASYNC109
    """Close all connection pools gracefully.

    This function closes all pools and waits for all connections to be
    released properly. If graceful shutdown takes too long, it will
    forcefully terminate the pools.

    Args:
        pools: Dictionary mapping database names to their pools.
        timeout: Maximum time in seconds to wait for graceful shutdown
            before forcing termination. Default: 10.0 seconds.

    Example:
        >>> pools = await create_pools(configs)
        >>> # ... use pools ...
        >>> await close_pools(pools, timeout=5.0)
    """
    for db_name, pool in pools.items():
        try:
            # Try graceful close with timeout
            await asyncio.wait_for(pool.close(), timeout=timeout)
            logger.info(f"Connection pool for '{db_name}' closed gracefully")
        except TimeoutError:
            # Force termination if graceful close times out
            logger.warning(
                f"Graceful close timed out for '{db_name}', forcing termination"
            )
            pool.terminate()
            logger.info(f"Connection pool for '{db_name}' terminated")
        except Exception as e:
            # Log error but continue closing other pools
            logger.error(f"Error closing pool for '{db_name}': {e!s}")
            # Force terminate on error
            pool.terminate()
