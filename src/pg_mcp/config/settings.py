"""Configuration management for PostgreSQL MCP Server.

This module defines all configuration settings using Pydantic for validation
and type safety. Configuration is loaded from environment variables with
sensible defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _find_env_file() -> str | None:
    """Locate the nearest .env starting from this package up to cwd."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            return str(candidate)
    cwd_env = Path.cwd() / ".env"
    return str(cwd_env) if cwd_env.is_file() else None


_ENV_FILE = _find_env_file()


def load_env_file() -> None:
    """Load `.env` into os.environ so nested configs can read OPENAI_*/DATABASE_*."""
    if not _ENV_FILE:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ENV_FILE, override=False)


def _parse_csv_or_list(value: str | list[str]) -> list[str]:
    """Parse a comma-separated string or pass through a list."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class DatabaseConfig(BaseSettings):
    """PostgreSQL database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_", extra="ignore")

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="postgres", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")
    ssl: bool | None = Field(
        default=None,
        description=(
            "Use SSL for PostgreSQL. None disables SSL for localhost "
            "and enables it for remote hosts."
        ),
    )

    # Connection pool settings
    min_pool_size: int = Field(default=5, ge=1, le=100, description="Minimum pool size")
    max_pool_size: int = Field(default=20, ge=1, le=100, description="Maximum pool size")
    pool_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Pool acquire timeout in seconds"
    )
    command_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Command execution timeout in seconds"
    )

    @property
    def dsn(self) -> str:
        """Build PostgreSQL DSN connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def safe_dsn(self) -> str:
        """Build DSN with masked password for logging."""
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.name}"

    @property
    def use_ssl(self) -> bool:
        """Whether the pool should require SSL."""
        if self.ssl is not None:
            return self.ssl
        return self.host not in {"localhost", "127.0.0.1", "::1"}


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_", extra="ignore")

    api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible API base URL (relay/proxy). None uses official OpenAI.",
    )
    model: str = Field(default="gpt-4o-mini", description="Model to use for SQL generation")
    max_tokens: int = Field(default=2000, ge=100, le=128000, description="Maximum tokens in response")
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Temperature for response randomness"
    )
    timeout: float = Field(
        default=30.0, ge=5.0, le=120.0, description="API request timeout in seconds"
    )

    @field_validator("base_url", mode="before")
    @classmethod
    def empty_base_url_to_none(cls, v: str | None) -> str | None:
        """Treat blank OPENAI_BASE_URL as unset (official OpenAI)."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        """Validate API key is not empty."""
        api_key_str = v.get_secret_value()
        if not api_key_str or not api_key_str.strip():
            raise ValueError("OpenAI API key must not be empty")
        return v

    @model_validator(mode="after")
    def validate_official_api_key_prefix(self) -> OpenAIConfig:
        """Official OpenAI keys must start with sk-; relays may use other prefixes."""
        if self.base_url is None:
            api_key_str = self.api_key.get_secret_value()
            if not api_key_str.startswith("sk-"):
                raise ValueError("OpenAI API key must start with 'sk-'")
        return self


class SecurityConfig(BaseSettings):
    """Security and access control configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_", extra="ignore")

    allow_write_operations: bool = Field(
        default=False, description="Allow write operations (INSERT, UPDATE, DELETE)"
    )
    blocked_functions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "pg_sleep",
            "pg_read_file",
            "pg_write_file",
            "lo_import",
            "lo_export",
        ],
        description="List of blocked PostgreSQL functions",
    )
    blocked_tables: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Table names that must not be queried",
    )
    blocked_columns: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Column names that must not be selected (name or table.column)",
    )
    allow_explain: bool = Field(
        default=False, description="Whether EXPLAIN statements are permitted"
    )
    max_rows: int = Field(default=10000, ge=1, le=100000, description="Maximum rows to return")
    max_execution_time: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Maximum query execution time in seconds"
    )
    readonly_role: str | None = Field(
        default=None, description="PostgreSQL role to switch to for read-only access"
    )
    safe_search_path: str = Field(
        default="public", description="Safe search_path to set during query execution"
    )

    @field_validator("blocked_functions", "blocked_tables", "blocked_columns", mode="before")
    @classmethod
    def parse_blocked_lists(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string or list."""
        return _parse_csv_or_list(v)


class ValidationConfig(BaseSettings):
    """Query validation configuration."""

    model_config = SettingsConfigDict(env_prefix="VALIDATION_", extra="ignore")

    max_question_length: int = Field(
        default=10000, ge=1, le=50000, description="Maximum question length in characters"
    )
    min_confidence_score: int = Field(
        default=70, ge=0, le=100, description="Minimum confidence score (0-100)"
    )

    # Result validation settings
    enabled: bool = Field(default=True, description="Enable result validation using LLM")
    sample_rows: int = Field(
        default=5, ge=1, le=100, description="Number of sample rows to include in validation"
    )
    timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Result validation timeout in seconds"
    )
    confidence_threshold: int = Field(
        default=70, ge=0, le=100, description="Minimum confidence for acceptable results"
    )


class CacheConfig(BaseSettings):
    """Schema cache configuration."""

    model_config = SettingsConfigDict(env_prefix="CACHE_", extra="ignore")

    schema_ttl: int = Field(
        default=3600, ge=60, le=86400, description="Schema cache TTL in seconds"
    )
    max_size: int = Field(default=100, ge=1, le=1000, description="Maximum cache entries")
    enabled: bool = Field(default=True, description="Enable schema caching")


class ResilienceConfig(BaseSettings):
    """Resilience and fault tolerance configuration."""

    model_config = SettingsConfigDict(env_prefix="RESILIENCE_", extra="ignore")

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Initial retry delay in seconds"
    )
    backoff_factor: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff factor"
    )
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, le=100, description="Failures before circuit opens"
    )
    circuit_breaker_timeout: float = Field(
        default=60.0, ge=10.0, le=300.0, description="Circuit breaker timeout in seconds"
    )
    max_concurrent_queries: int = Field(
        default=10, ge=1, le=1000, description="Maximum concurrent database queries"
    )
    max_concurrent_llm_calls: int = Field(
        default=5, ge=1, le=1000, description="Maximum concurrent LLM API calls"
    )


class ObservabilityConfig(BaseSettings):
    """Observability and monitoring configuration."""

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_", extra="ignore")

    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(
        default=9090, ge=1024, le=65535, description="Metrics HTTP server port"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(default="text", description="Log format")


class Settings(BaseSettings):
    """Main application settings aggregating all config sections."""

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )

    # Nested configurations
    databases: list[DatabaseConfig] = Field(
        default_factory=list, description="Configured PostgreSQL databases"
    )
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @field_validator("databases", mode="before")
    @classmethod
    def parse_databases(cls, v: Any) -> Any:
        """Parse DATABASES JSON string from the environment."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    @model_validator(mode="before")
    @classmethod
    def coerce_legacy_database(cls, data: Any) -> Any:
        """Accept legacy `database=` input and DATABASES env JSON."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        legacy = data.pop("database", None)
        if legacy is not None:
            data["databases"] = [legacy] if not isinstance(legacy, list) else legacy
            return data
        if data.get("databases"):
            return data
        raw = os.environ.get("DATABASES")
        if raw and raw.strip():
            try:
                data["databases"] = json.loads(raw)
            except json.JSONDecodeError:
                pass
        return data

    @model_validator(mode="after")
    def validate_databases(self) -> Settings:
        """Ensure at least one uniquely named database is configured."""
        if not self.databases:
            self.databases = [DatabaseConfig()]
        names = [db.name for db in self.databases]
        if len(names) != len(set(names)):
            raise ValueError("Database names must be unique")
        return self

    @property
    def database(self) -> DatabaseConfig:
        """Primary (first) database, kept for backward compatibility."""
        return self.databases[0]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance.

    Returns:
        Settings: The global settings instance.
    """
    global _settings
    if _settings is None:
        load_env_file()
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset global settings instance. Useful for testing."""
    global _settings
    _settings = None
