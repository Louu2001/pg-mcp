"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements retry logic, error handling, and request tracking.
"""

import logging
import time
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseConnectionError,
    DatabaseError,
    ErrorCode,
    LLMError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import request_context
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.resilience.retry import retry_async
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)

_TRANSIENT_LLM_ERRORS: tuple[type[BaseException], ...] = ()
_TRANSIENT_DB_ERRORS = (DatabaseConnectionError,)
_RATE_LIMIT_TIMEOUT_SECONDS = 30.0


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation. It implements retry logic with error feedback, circuit breaker
    pattern for fault tolerance, and comprehensive error handling.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executor=executor,
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        sql_executors: dict[str, SQLExecutor] | None = None,
        sql_executor: SQLExecutor | None = None,
        rate_limiter: MultiRateLimiter | None = None,
        metrics: MetricsCollector | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries and circuit breaker.
            validation_config: Validation configuration including thresholds.
            sql_executors: Per-database SQL execution services.
            sql_executor: Legacy single executor used when ``sql_executors`` is omitted.
            rate_limiter: Optional limiter wrapping LLM and DB calls.
            metrics: Optional Prometheus metrics collector.
            circuit_breaker: Shared circuit breaker; created from config when omitted.
        """
        if sql_executors is None:
            if sql_executor is None:
                raise TypeError("sql_executors or sql_executor is required")
            sql_executors = dict.fromkeys(pools, sql_executor)

        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.sql_executors = sql_executors
        if sql_executor is not None:
            self.sql_executor = sql_executor
        elif sql_executors:
            self.sql_executor = next(iter(sql_executors.values()))
        else:
            raise TypeError("sql_executors or sql_executor is required")
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.rate_limiter = rate_limiter
        self.metrics = metrics
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline:
        1. Generate request_id for tracking
        2. Resolve and validate database name
        3. Load schema from cache
        4. Generate and validate SQL with retry logic
        5. Execute SQL (if return_type == RESULT)
        6. Validate results (optional)
        7. Return structured response

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.
        """
        async with request_context() as request_id:
            return await self._execute_query(request, request_id)

    async def _execute_query(self, request: QueryRequest, request_id: str) -> QueryResponse:
        """Run the query pipeline inside a tracing context."""
        logger.info(
            "Starting query execution",
            extra={"request_id": request_id, "question": request.question[:100]},
        )
        started = time.perf_counter()
        database_name = "unknown"
        status = "error"

        try:
            self._enforce_question_length(request.question)
            self._reject_write_intent(request.question)

            database_name = self._resolve_database(request.database)
            logger.debug(
                "Resolved database",
                extra={"request_id": request_id, "database": database_name},
            )

            schema = self.schema_cache.get(database_name)
            if schema is None:
                pool = self.pools.get(database_name)
                if pool is None:
                    raise DatabaseError(
                        message=f"No connection pool available for database '{database_name}'",
                        details={"database": database_name},
                    )
                try:
                    schema = await self.schema_cache.load(database_name, pool)
                except Exception as e:
                    raise SchemaLoadError(
                        message=f"Failed to load schema for database '{database_name}': {e!s}",
                        details={"database": database_name, "error": str(e)},
                    ) from e

            logger.debug(
                "Schema loaded",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "tables": len(schema.tables),
                },
            )

            schema = self._schema_for_generation(schema)
            generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
                question=request.question,
                schema=schema,
                request_id=request_id,
            )

            if request.return_type == ReturnType.SQL:
                logger.info(
                    "Returning SQL only",
                    extra={"request_id": request_id, "sql_length": len(generated_sql)},
                )
                status = "success"
                return QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=None,
                    error=None,
                    confidence=100,
                    tokens_used=tokens_used or 0,
                )

            logger.debug("Executing SQL", extra={"request_id": request_id})
            start_time = self._get_current_time_ms()
            results, total_count = await self._execute_sql(database_name, generated_sql)
            execution_time_ms = self._get_current_time_ms() - start_time
            logger.info(
                "SQL executed successfully",
                extra={
                    "request_id": request_id,
                    "row_count": total_count,
                    "execution_time_ms": execution_time_ms,
                },
            )

            result_confidence = await self._validate_results_safely(
                question=request.question,
                sql=generated_sql,
                results=results,
                row_count=total_count,
                request_id=request_id,
            )

            query_result = QueryResult(
                columns=list(results[0].keys()) if results else [],
                rows=results,
                row_count=len(results),
                execution_time_ms=execution_time_ms,
            )
            status = "success"
            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=query_result,
                error=None,
                confidence=result_confidence,
                tokens_used=tokens_used or 0,
            )

        except PgMcpError as e:
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            status = e.code.value if isinstance(e.code, ErrorCode) else str(e.code)
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=e.code.value if isinstance(e.code, ErrorCode) else str(e.code),
                    message=e.message,
                    details=e.details,
                ),
                confidence=0,
                tokens_used=0,
            )
        except Exception as e:
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id},
            )
            status = "internal_error"
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message=f"Internal server error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
                confidence=0,
                tokens_used=0,
            )
        finally:
            self._record_query_metrics(status=status, database=database_name, started=started)

    def _schema_for_generation(self, schema: Any) -> Any:
        """Hide blocked tables/columns from the LLM prompt."""
        from pg_mcp.models.schema import DatabaseSchema, TableInfo

        raw_tables = getattr(self.sql_validator, "blocked_tables", None)
        raw_columns = getattr(self.sql_validator, "blocked_columns", None)
        blocked_tables = (
            {str(name).lower() for name in raw_tables}
            if isinstance(raw_tables, (set, list, frozenset))
            else set()
        )
        blocked_columns = (
            {str(name).lower() for name in raw_columns}
            if isinstance(raw_columns, (set, list, frozenset))
            else set()
        )
        if not isinstance(schema, DatabaseSchema) or (not blocked_tables and not blocked_columns):
            return schema

        filtered: list[TableInfo] = []
        for table in schema.tables:
            if table.table_name.lower() in blocked_tables:
                continue
            columns = [
                col
                for col in table.columns
                if col.name.lower() not in blocked_columns
                and f"{table.table_name.lower()}.{col.name.lower()}" not in blocked_columns
            ]
            filtered.append(table.model_copy(update={"columns": columns}))
        return schema.model_copy(update={"tables": filtered})

    @staticmethod
    def _is_hard_security_violation(error: SecurityViolationError) -> bool:
        """Blocked tables/columns must fail closed without LLM rewrite retries."""
        message = str(error).lower()
        return (
            "table" in message
            or "column" in message
            or "不支持数据修改" in str(error)
            or "write" in message
        )

    def _enforce_question_length(self, question: str) -> None:
        """Reject questions that exceed the configured length."""
        max_length = self.validation_config.max_question_length
        if len(question) > max_length:
            raise PgMcpError(
                message=f"Question exceeds maximum length of {max_length} characters",
                code=ErrorCode.QUESTION_TOO_LONG,
                details={"length": len(question), "max_length": max_length},
            )

    def _reject_write_intent(self, question: str) -> None:
        """Fail closed on natural-language write/DDL requests before SQL generation."""
        check_intent = getattr(self.sql_validator, "check_question_intent", None)
        if not callable(check_intent):
            return
        try:
            check_intent(question)
        except SecurityViolationError as exc:
            self._record_sql_rejected(exc)
            raise

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.
        """
        available_dbs = list(self.pools.keys()) or list(self.sql_executors.keys())
        if database is not None:
            if database not in self.pools and database not in self.sql_executors:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": available_dbs,
                    },
                )
            return database

        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    def _get_executor(self, database_name: str) -> SQLExecutor:
        """Return the executor bound to ``database_name``."""
        executor = self.sql_executors.get(database_name)
        if executor is None:
            raise DatabaseError(
                message=f"No SQL executor available for database '{database_name}'",
                details={
                    "database": database_name,
                    "available_databases": list(self.sql_executors.keys()),
                },
            )
        return executor

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures."""
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                generated_sql = await self._call_llm_generate(
                    question=question,
                    schema=schema,
                    previous_sql=previous_sql,
                    error_feedback=error_feedback,
                )
                tokens_used = getattr(self.sql_generator, "last_tokens_used", None)
                if self.metrics and isinstance(tokens_used, int) and tokens_used > 0:
                    self.metrics.increment_llm_tokens("generate_sql", tokens_used)

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except SecurityViolationError as validation_error:
                    self._record_sql_rejected(validation_error)
                    if self._is_hard_security_violation(validation_error):
                        raise
                    if attempt < max_retries:
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        continue
                    self.circuit_breaker.record_failure()
                    raise
                except SQLParseError as validation_error:
                    self._record_sql_rejected(validation_error)
                    if attempt < max_retries:
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        continue
                    self.circuit_breaker.record_failure()
                    logger.error(
                        "SQL validation failed after all retries",
                        extra={
                            "request_id": request_id,
                            "attempts": attempt + 1,
                            "error": str(validation_error),
                        },
                    )
                    raise

                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )
                return generated_sql, validation_result, tokens_used

            except (LLMError, SecurityViolationError, SQLParseError, RateLimitExceededError):
                raise
            except Exception as e:
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _call_llm_generate(
        self,
        question: str,
        schema: Any,
        previous_sql: str | None,
        error_feedback: str | None,
    ) -> str:
        """Generate SQL with rate limiting and transient retries."""

        async def _generate() -> str:
            started = time.perf_counter()
            sql = await self.sql_generator.generate(
                question=question,
                schema=schema,
                previous_attempt=previous_sql,
                error_feedback=error_feedback,
            )
            if self.metrics:
                self.metrics.increment_llm_call("generate_sql")
                self.metrics.observe_llm_latency("generate_sql", time.perf_counter() - started)
            return sql

        return await self._run_with_limit_and_retry(
            _generate,
            limiter="llm",
            retryable=_TRANSIENT_LLM_ERRORS,
        )

    async def _execute_sql(
        self,
        database_name: str,
        sql: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """Execute SQL against the resolved database with limits and retries."""
        executor = self._get_executor(database_name)

        async def _execute() -> tuple[list[dict[str, Any]], int]:
            started = time.perf_counter()
            result = await executor.execute(sql)
            if self.metrics:
                self.metrics.observe_db_query_duration(time.perf_counter() - started)
            return result

        return await self._run_with_limit_and_retry(
            _execute,
            limiter="query",
            retryable=_TRANSIENT_DB_ERRORS,
        )

    async def _run_with_limit_and_retry(
        self,
        operation: Any,
        *,
        limiter: str,
        retryable: tuple[type[BaseException], ...],
    ) -> Any:
        """Apply rate limiting and exponential backoff around an operation."""
        max_attempts = max(1, self.resilience_config.max_retries)

        async def _retry() -> Any:
            return await retry_async(
                operation,
                max_attempts=max_attempts,
                initial_delay=self.resilience_config.retry_delay,
                backoff_factor=self.resilience_config.backoff_factor,
                retryable=retryable,
            )

        if self.rate_limiter is None:
            return await _retry()

        try:
            if limiter == "llm":
                async with self.rate_limiter.for_llm(timeout=_RATE_LIMIT_TIMEOUT_SECONDS):
                    return await _retry()
            async with self.rate_limiter.for_queries(timeout=_RATE_LIMIT_TIMEOUT_SECONDS):
                return await _retry()
        except TimeoutError as exc:
            raise RateLimitExceededError(
                message="Rate limit exceeded; too many concurrent requests",
                details={"limiter": limiter, "timeout": _RATE_LIMIT_TIMEOUT_SECONDS},
            ) from exc

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking)."""
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            started = time.perf_counter()
            validation_result = await self.result_validator.validate(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
            )
            if self.metrics:
                self.metrics.increment_llm_call("validate_result")
                self.metrics.observe_llm_latency("validate_result", time.perf_counter() - started)

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                    "threshold": self.validation_config.min_confidence_score,
                },
            )

            return validation_result.confidence

        except Exception as e:
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 0

    def _record_sql_rejected(self, error: Exception) -> None:
        """Emit a metric for a rejected SQL statement."""
        if self.metrics is None:
            return
        reason = (
            "security_violation" if isinstance(error, SecurityViolationError) else "parse_error"
        )
        self.metrics.increment_sql_rejected(reason)

    def _record_query_metrics(self, *, status: str, database: str, started: float) -> None:
        """Record request-level duration and count."""
        if self.metrics is None:
            return
        self.metrics.query_duration.observe(time.perf_counter() - started)
        self.metrics.increment_query_request(status=status, database=database)

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds."""
        return time.time() * 1000
