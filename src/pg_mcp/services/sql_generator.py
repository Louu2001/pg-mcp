"""SQL generation service using OpenAI for natural language to SQL conversion.

This module provides the SQLGenerator class that uses OpenAI's LLM to convert
natural language questions into valid PostgreSQL SQL queries.
"""

import asyncio
import re
from typing import TYPE_CHECKING, Any

from openai import (
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from pg_mcp.config.settings import OpenAIConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.prompts.sql_generation import SQL_GENERATION_SYSTEM_PROMPT, build_user_prompt

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

    from pg_mcp.models.schema import DatabaseSchema


class SQLGenerator:
    """SQL generator using OpenAI for natural language to SQL conversion.

    This class handles the interaction with OpenAI's API to generate SQL queries
    from natural language questions. It includes robust error handling, SQL extraction
    from various response formats, and support for retry scenarios with error feedback.

    Example:
        >>> config = OpenAIConfig(api_key="sk-...", model="gpt-4")
        >>> generator = SQLGenerator(config)
        >>> sql = await generator.generate(
        ...     question="How many users registered today?",
        ...     schema=db_schema
        ... )
    """

    def __init__(self, config: OpenAIConfig) -> None:
        """Initialize SQL generator with OpenAI configuration.

        Args:
            config: OpenAI configuration including API key and model settings.
        """
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout,
        )
        self.last_tokens_used = 0

    async def generate(
        self,
        question: str,
        schema: "DatabaseSchema",
        context: str | None = None,
        previous_attempt: str | None = None,
        error_feedback: str | None = None,
    ) -> str:
        """Generate SQL statement from natural language question.

        This method sends the question and database schema to OpenAI's API
        and extracts the generated SQL query from the response. It supports
        retry scenarios by accepting previous failed attempts and error feedback.

        Args:
            question: User's natural language question.
            schema: Database schema information for context.
            context: Optional additional context to guide generation.
            previous_attempt: Previously generated SQL that failed (for retry).
            error_feedback: Error message from previous attempt (for retry).

        Returns:
            str: Generated SQL query (without trailing semicolon).

        Raises:
            LLMError: If generation fails or response is invalid.
            LLMTimeoutError: If the API request times out.
            LLMUnavailableError: If the API is unavailable or authentication fails.

        Example:
            >>> # Initial generation
            >>> sql = await generator.generate(
            ...     question="Count all active users",
            ...     schema=db_schema
            ... )
            >>> # Retry with error feedback
            >>> sql = await generator.generate(
            ...     question="Count all active users",
            ...     schema=db_schema,
            ...     previous_attempt="SELECT COUNT(*) FROM user",
            ...     error_feedback='relation "user" does not exist'
            ... )
        """
        user_prompt = build_user_prompt(
            question=question,
            schema=schema,
            context=context,
            previous_attempt=previous_attempt,
            error_feedback=error_feedback,
        )
        request_kwargs = self._completion_kwargs(user_prompt)

        try:
            async with asyncio.timeout(self.config.timeout):
                response: ChatCompletion = await self.client.chat.completions.create(
                    **request_kwargs
                )
        except (TimeoutError, APITimeoutError) as e:
            raise LLMTimeoutError(
                message=(
                    f"LLM 请求超时（{self.config.timeout}s）。"
                    "当前中转站对 gpt-5.5 可能无响应，请换可用模型或检查订阅。"
                ),
                details={"timeout": self.config.timeout},
            ) from e
        except AuthenticationError as e:
            raise LLMUnavailableError(
                message="OpenAI API 认证失败，请检查 API Key",
                details={"error": str(e)},
            ) from e
        except PermissionDeniedError as e:
            raise LLMUnavailableError(
                message="LLM 服务拒绝访问：中转站余额不足或未开通订阅",
                details={"error": str(e)},
            ) from e
        except RateLimitError as e:
            raise LLMUnavailableError(
                message="OpenAI API 触发限流，请稍后重试",
                details={"error": str(e)},
            ) from e
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                raise LLMUnavailableError(
                    message="OpenAI API authentication failed - check API key",
                    details={"error": error_msg},
                ) from e
            if "rate_limit" in error_msg.lower():
                raise LLMUnavailableError(
                    message="OpenAI API rate limit exceeded",
                    details={"error": error_msg},
                ) from e
            raise LLMError(
                message=f"OpenAI API request failed: {error_msg}",
                details={"error": error_msg},
            ) from e

        # Extract SQL from response
        if not response.choices:
            raise LLMError(
                message="OpenAI returned empty response",
                details={"response": response.model_dump()},
            )

        content = response.choices[0].message.content
        if not content:
            raise LLMError(
                message="OpenAI returned empty message content",
                details={"response": response.model_dump()},
            )

        sql = self._extract_sql(content)
        if not sql:
            raise LLMError(
                message="Failed to extract SQL from OpenAI response",
                details={"content": content},
            )

        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        self.last_tokens_used = total_tokens if isinstance(total_tokens, int) else 0

        return sql

    def _completion_kwargs(self, user_prompt: str) -> dict[str, Any]:
        """Build chat.completions parameters for the configured model."""
        tokens = min(self.config.max_tokens, 2048)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SQL_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        model = self.config.model.lower()
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["max_completion_tokens"] = tokens
        else:
            kwargs["temperature"] = self.config.temperature
            kwargs["max_tokens"] = tokens
        return kwargs

    def _extract_sql(self, content: str) -> str | None:
        """Extract SQL query from LLM response content.

        This method implements a multi-strategy approach to extract SQL from
        various response formats that LLMs might generate:

        1. Try to match ```sql ... ``` code blocks (preferred format)
        2. Try to match generic ``` ... ``` code blocks
        3. Try to find SELECT/WITH statements in plain text
        4. Check if entire content looks like SQL

        Args:
            content: Raw content from LLM response.

        Returns:
            str | None: Extracted SQL query, or None if extraction fails.

        Example:
            >>> generator._extract_sql("```sql\\nSELECT 1;\\n```")
            'SELECT 1;'
            >>> generator._extract_sql("SELECT * FROM users;")
            'SELECT * FROM users;'
        """
        if not content:
            return None

        content = content.strip()

        # Strategy 1: Match ```sql ... ``` or ``` ... ``` code blocks
        code_block_pattern = r"```(?:sql)?\s*\n?(.*?)\n?```"
        matches = re.findall(code_block_pattern, content, re.DOTALL | re.IGNORECASE)

        if matches:
            sql = matches[0].strip()
            # Remove trailing semicolon for consistency
            return sql.rstrip(";") + ";"

        # Strategy 2: Find SELECT/WITH statements in plain text
        sql_pattern = r"((?:WITH|SELECT)\s+.*?)(?:;|$)"
        matches = re.findall(sql_pattern, content, re.DOTALL | re.IGNORECASE)

        if matches:
            sql = matches[0].strip()
            return sql.rstrip(";") + ";"

        # Strategy 3: Check if entire content looks like SQL
        if content.upper().startswith(("SELECT", "WITH")):
            return content.rstrip(";") + ";"

        return None
