"""Unit tests for judge model extra parameters functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from judge.llm_judge import LLMJudge


@pytest.mark.unit
class TestJudgeExtraParams:
    """Test that extra parameters are properly passed through the judge system."""

    def test_llm_judge_accepts_extra_params(self, fixtures_dir: Path):
        """Test that LLMJudge accepts judge_model_extra_params parameter."""
        extra_params = {"temperature": 0.7, "max_tokens": 1000}

        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        assert judge.judge_model_extra_params == extra_params

    def test_llm_judge_extra_params_defaults_to_empty_dict(self, fixtures_dir: Path):
        """Test that judge_model_extra_params defaults to empty dict."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        assert judge.judge_model_extra_params == {}

    def test_llm_judge_stores_extra_params_correctly(self, fixtures_dir: Path):
        """Test that LLMJudge stores extra params and makes them available."""
        extra_params = {"temperature": 0.5, "max_tokens": 500, "top_p": 0.9}

        judge = LLMJudge(
            judge_model="claude-3-7-sonnet",
            judge_model_extra_params=extra_params,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Verify extra params are stored correctly
        assert judge.judge_model_extra_params == extra_params
        assert judge.judge_model_extra_params["temperature"] == 0.5
        assert judge.judge_model_extra_params["max_tokens"] == 500
        assert judge.judge_model_extra_params["top_p"] == 0.9

        # Verify standard model param is still accessible
        assert judge.judge_model == "claude-3-7-sonnet"

    @pytest.mark.skip(
        reason="Mocking LLMFactory in async flow is complex; covered by runner tests"
    )
    @pytest.mark.asyncio
    async def test_llm_judge_passes_extra_params_in_async_evaluation(
        self, fixtures_dir: Path
    ):
        """Test that extra params are passed to LLMFactory during async evaluation."""
        extra_params = {"temperature": 0.7, "max_tokens": 1000}

        # Patch at both the source and the import location
        with patch("judge.llm_judge.LLMFactory") as mock_factory_class:
            from judge.response_models import QuestionResponse

            # Create a mock LLM that supports structured output
            mock_llm = MagicMock()
            mock_llm.__class__.__name__ = "ClaudeLLM"

            # Mock the generate_structured method to return a proper response
            mock_response = QuestionResponse(
                answer="No", reasoning="Test reasoning", question_text="Test question"
            )
            mock_llm.generate_structured = AsyncMock(return_value=mock_response)

            # Setup the factory mock
            mock_factory_class.create_llm.return_value = mock_llm

            judge = LLMJudge(
                judge_model="claude-3-7-sonnet",
                judge_model_extra_params=extra_params,
                rubric_folder=str(fixtures_dir),
                rubric_file="rubric_simple.tsv",
                rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            )

            # Create a simple conversation for testing
            conversation_file = fixtures_dir / "test_conversation.txt"
            conversation_file.write_text("User: Hello\nAssistant: Hi there!")

            # Run async evaluation - this will trigger LLM creation
            try:
                await judge.evaluate_conversation_question_flow(
                    str(conversation_file),
                    output_folder=str(fixtures_dir),
                    auto_save=False,
                )
            except (KeyError, AttributeError, TypeError):
                # Expected errors from mock navigation or incomplete evaluation
                # We only care that create_llm was called correctly
                pass

            # Verify create_llm was called with extra params
            assert (
                mock_factory_class.create_llm.called
            ), "LLMFactory.create_llm should be called during evaluation"
            call_kwargs = mock_factory_class.create_llm.call_args[1]
            assert (
                "temperature" in call_kwargs
            ), f"Expected temperature in {call_kwargs}"
            assert call_kwargs["temperature"] == 0.7
            assert "max_tokens" in call_kwargs, f"Expected max_tokens in {call_kwargs}"
            assert call_kwargs["max_tokens"] == 1000

    def test_llm_judge_extra_params_with_none(self, fixtures_dir: Path):
        """Test that passing None for extra_params works correctly."""
        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=None,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        assert judge.judge_model_extra_params == {}

    def test_llm_judge_preserves_standard_params(self, fixtures_dir: Path):
        """Test that extra params don't interfere with standard parameters."""
        extra_params = {"temperature": 0.8}

        judge = LLMJudge(
            judge_model="claude-3-7-sonnet",
            judge_model_extra_params=extra_params,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Standard params should still be accessible
        assert judge.judge_model == "claude-3-7-sonnet"
        assert judge.judge_model_extra_params == extra_params

    def test_multiple_extra_params_types(self, fixtures_dir: Path):
        """Test that extra params can contain various types."""
        extra_params = {
            "temperature": 0.7,  # float
            "max_tokens": 1000,  # int
            "top_p": 0.95,  # float
            "stop_sequences": ["END"],  # list
        }

        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        assert judge.judge_model_extra_params == extra_params
        assert isinstance(judge.judge_model_extra_params["temperature"], float)
        assert isinstance(judge.judge_model_extra_params["max_tokens"], int)
        assert isinstance(judge.judge_model_extra_params["stop_sequences"], list)
