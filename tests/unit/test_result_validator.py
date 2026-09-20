"""Unit tests for ResultValidator with a mocked OpenAI client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from pg_mcp.config.settings import OpenAIConfig, ValidationConfig
from pg_mcp.models.errors import LLMError
from pg_mcp.services.result_validator import ResultValidator


def _validator(
    *,
    enabled: bool = True,
    threshold: int = 70,
) -> ResultValidator:
    return ResultValidator(
        openai_config=OpenAIConfig(api_key=SecretStr("sk-test-key-12345")),
        validation_config=ValidationConfig(
            enabled=enabled,
            confidence_threshold=threshold,
            min_confidence_score=threshold,
        ),
    )


class TestResultValidator:
    """Result validation scoring and error handling."""

    @pytest.mark.asyncio
    async def test_disabled_returns_high_confidence(self) -> None:
        """Disabled validation should skip the LLM and return 100."""
        validator = _validator(enabled=False)
        result = await validator.validate(
            question="How many users?",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 3}],
            row_count=1,
        )
        assert result.confidence == 100
        assert result.is_acceptable is True

    @pytest.mark.asyncio
    async def test_parses_llm_json_and_applies_threshold(self) -> None:
        """Confidence from the LLM is compared to the configured threshold."""
        validator = _validator(threshold=80)
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"confidence": 65, "explanation": "weak match", "suggestion": null}'
                )
            )
        ]

        with patch.object(
            validator.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await validator.validate(
                question="How many users?",
                sql="SELECT COUNT(*) FROM users",
                results=[{"count": 3}],
                row_count=1,
            )

        assert result.confidence == 65
        assert result.is_acceptable is False
        assert result.explanation == "weak match"

    @pytest.mark.asyncio
    async def test_empty_llm_response_raises(self) -> None:
        """Empty choices should surface as an LLM error."""
        validator = _validator()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_response.model_dump.return_value = {}

        with (
            patch.object(
                validator.client.chat.completions,
                "create",
                new=AsyncMock(return_value=mock_response),
            ),
            pytest.raises(LLMError),
        ):
            await validator.validate(
                question="How many users?",
                sql="SELECT 1",
                results=[{"n": 1}],
                row_count=1,
            )
