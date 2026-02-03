"""Unit tests for judge.py CLI argument parsing."""

import argparse

from judge.utils import parse_judge_models


def _setup_judge_model_arg(argv: list[str]) -> list[str]:
    """Parse argv and return args.judge_model (same type as judge.py CLI)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-model",
        "-j",
        nargs="+",
        required=True,
        help="Model(s) to use for judging; format 'model' or 'model:count'",
    )
    args = parser.parse_args(argv)
    return args.judge_model


class TestJudgeModelParsing:
    """Test parsing of --judge-model CLI argument (same nargs='+' list as judge.py)."""

    def test_single_model(self):
        """Test parsing a single model without count."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 1}

    def test_single_model_with_count(self):
        """Test parsing a single model with count."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:3"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 3}

    def test_multiple_different_models(self):
        """Test parsing multiple different models."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o", "claude-sonnet-4-5-20250929"]
        )
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 1, "claude-sonnet-4-5-20250929": 1}

    def test_multiple_models_with_counts(self):
        """Test parsing multiple models with counts."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o:2", "claude-sonnet-4-5-20250929:3"]
        )
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 2, "claude-sonnet-4-5-20250929": 3}

    def test_mixed_models_with_and_without_counts(self):
        """Test parsing mix of models with and without counts."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o", "claude-sonnet-4-5-20250929:2"]
        )
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 1, "claude-sonnet-4-5-20250929": 2}

    def test_model_with_multiple_colons(self):
        """Test parsing ollama-style model with colon in name (e.g. llama:7b:3)."""
        judge_model = _setup_judge_model_arg(["-j", "llama:7b:3"])
        result = parse_judge_models(judge_model)
        assert result == {"llama:7b": 3}

    def test_three_models_mixed(self):
        """Test parsing three models with various count specifications."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o:2", "claude-sonnet-4-5-20250929", "gpt-3.5-turbo:3"]
        )
        result = parse_judge_models(judge_model)
        assert result == {
            "gpt-4o": 2,
            "claude-sonnet-4-5-20250929": 1,
            "gpt-3.5-turbo": 3,
        }

    def test_large_count(self):
        """Test parsing with large instance count."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:100"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 100}

    def test_duplicate_models_last_wins(self):
        """Test that if same model specified twice, last value wins."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:2", "gpt-4o:5"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 5}
