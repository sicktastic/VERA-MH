"""Async unit tests for LLMJudge question flow navigation logic."""

from pathlib import Path

import pytest

from judge.llm_judge import LLMJudge


@pytest.mark.unit
class TestEvaluateQuestionFlow:
    """Test the _evaluate_question_flow method for navigation logic."""

    @pytest.mark.asyncio
    async def test_evaluate_question_flow_basic(self, fixtures_dir: Path):
        """Test basic question flow navigation from start to end."""
        from tests.mocks.mock_llm import MockLLM

        mock_responses = [
            "ANSWER: Yes\nREASONING: The conversation shows clear indicators.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_single_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        result = await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="1",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Verify navigation completed without early stopping
        assert result is None

        # Verify dimension answers were tracked
        assert "Dimension A" in dimension_answers
        assert len(dimension_answers["Dimension A"]) == 1
        assert dimension_answers["Dimension A"][0]["answer"] == "Yes"

    @pytest.mark.asyncio
    async def test_goto_branching(self, fixtures_dir: Path):
        """Test GOTO navigation to specific questions."""
        from tests.mocks.mock_llm import MockLLM

        mock_responses = [
            "ANSWER: Go to 3\nREASONING: Choosing path to question 3.",
            "ANSWER: Yes\nREASONING: Answering question 3.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_goto_logic.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        result = await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="1",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Verify it followed the GOTO path
        assert result is None
        assert len(evaluator.calls) == 2

        # Verify both dimensions tracked
        assert "Dimension A" in dimension_answers
        assert "Dimension B" in dimension_answers

    @pytest.mark.asyncio
    async def test_goto_end_stops_evaluation(self, fixtures_dir: Path):
        """Test GOTO: END stops the evaluation flow."""
        from tests.mocks.mock_llm import MockLLM

        mock_responses = [
            "ANSWER: Done\nREASONING: Ending evaluation.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_goto_logic.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        result = await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="2",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Verify evaluation stopped
        assert result is None
        assert len(evaluator.calls) == 1

        # Only Dimension A should be tracked
        assert "Dimension A" in dimension_answers
        assert "Dimension B" not in dimension_answers

    @pytest.mark.asyncio
    async def test_early_stopping_on_not_relevant_q4(self, fixtures_dir: Path):
        """Test early stopping when question 4 returns 'Not Relevant'."""
        from tests.mocks.mock_llm import MockLLM

        rubric_content = (
            "Question ID\tDimension\tSeverity\tRisk Type\t"
            "Question\tExamples\tAnswer\tGOTO\n"
            "4\tDimension A\t\t\t"
            "Is this conversation relevant?\tTest example\t"
            "Not Relevant\tEND\n"
            "\t\t\t\t\t\tRelevant\t5\n"
            "5\tDimension B\t\t\t"
            "Follow-up question?\tTest example\tYes\tEND\n"
        )
        rubric_file = fixtures_dir / "rubric_early_stop.tsv"
        rubric_file.write_text(rubric_content)

        mock_responses = [
            "ANSWER: Not Relevant\nREASONING: Conversation is not relevant.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_early_stop.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        result = await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="4",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Verify early stopping occurred
        assert result == "4"
        assert len(evaluator.calls) == 1

    @pytest.mark.asyncio
    async def test_early_stopping_on_no_at_q6(self, fixtures_dir: Path):
        """Test early stopping when question 6 returns 'No'."""
        from tests.mocks.mock_llm import MockLLM

        rubric_content = (
            "Question ID\tDimension\tSeverity\tRisk Type\t"
            "Question\tExamples\tAnswer\tGOTO\n"
            "6\tDimension A\t\t\t"
            "Does conversation meet criteria?\tTest example\tYes\t7\n"
            "\t\t\t\t\t\tNo\t8\n"
            "7\tDimension B\t\t\t"
            "Follow-up question?\tTest example\tYes\tEND\n"
            "8\tDimension C\t\t\t"
            "Alternate question?\tTest example\tYes\tEND\n"
        )
        rubric_file = fixtures_dir / "rubric_q6_stop.tsv"
        rubric_file.write_text(rubric_content)

        mock_responses = [
            "ANSWER: No\nREASONING: Criteria not met.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_q6_stop.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        result = await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="6",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Verify early stopping occurred
        assert result == "6"

    @pytest.mark.asyncio
    async def test_question_flow_with_multiple_dimensions(self, fixtures_dir: Path):
        """Test question flow navigating through multiple dimensions."""
        from tests.mocks.mock_llm import MockLLM

        mock_responses = [
            "ANSWER: Yes\nREASONING: Dimension A criteria met.",
            "ANSWER: Yes\nREASONING: Dimension B criteria met.",
            "ANSWER: Yes\nREASONING: Dimension C criteria met.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_severity_values.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        result = await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="1",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Verify all dimensions were evaluated
        assert result is None
        assert "Dimension A" in dimension_answers
        assert "Dimension B" in dimension_answers
        assert "Dimension C" in dimension_answers

        # Verify all questions were answered "Yes"
        for dim_answers in dimension_answers.values():
            assert len(dim_answers) >= 1
            assert dim_answers[0]["answer"] == "Yes"

    @pytest.mark.asyncio
    async def test_answer_extraction_from_response(self, fixtures_dir: Path):
        """Test answer extraction from various LLM response formats."""
        from tests.mocks.mock_llm import MockLLM

        test_cases = [
            ("ANSWER: Yes\nREASONING: Clear yes.", "Yes"),
            ("ANSWER: No\nREASONING: Clear no.", "No"),
        ]

        for response_text, expected_answer in test_cases:
            mock_responses = [response_text]

            judge = LLMJudge(
                judge_model="mock-llm",
                rubric_folder=str(fixtures_dir),
                rubric_file="rubric_single_row.tsv",
                rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            )

            evaluator = MockLLM(responses=mock_responses)
            evaluator.set_system_prompt("Test system prompt")
            dimension_answers = {}

            await judge._evaluate_question_flow(
                evaluator=evaluator,
                start_question_id="1",
                dimension_answers=dimension_answers,
                verbose=False,
            )

            # Verify answer was extracted correctly
            if "Dimension A" in dimension_answers:
                extracted = dimension_answers["Dimension A"][0]["answer"]
                assert extracted in ["Yes", "No"]

    @pytest.mark.asyncio
    async def test_flow_follows_answer_path(self, fixtures_dir: Path):
        """Test that flow correctly follows the path based on answers."""
        from tests.mocks.mock_llm import MockLLM

        mock_responses_path_a = [
            "ANSWER: Option A\nREASONING: Taking path A.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_multi_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses_path_a)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="1",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Verify the path was followed
        assert "Dimension A" in dimension_answers
        assert dimension_answers["Dimension A"][0]["answer"] == "Option A"

    @pytest.mark.asyncio
    async def test_default_yes_moves_to_next_dimension(self, fixtures_dir: Path):
        """Test that answering 'Yes' without explicit GOTO moves to next dimension."""
        from tests.mocks.mock_llm import MockLLM

        rubric_content = (
            "Question ID\tDimension\tSeverity\tRisk Type\t"
            "Question\tExamples\tAnswer\tGOTO\n"
            "1\tDimension A\tRed\t\tFirst question?\tTest example\tYes\t\n"
            "\t\t\t\t\t\tNo\t2\n"
            "2\tDimension A\tYellow\t\t"
            "Second question?\tTest example\tYes\t\n"
            "3\tDimension B\t\t\tThird question?\tTest example\tYes\tEND\n"
        )
        rubric_file = fixtures_dir / "rubric_default_yes.tsv"
        rubric_file.write_text(rubric_content)

        mock_responses = [
            "ANSWER: Yes\nREASONING: Answering yes to Q1.",
            "ANSWER: Yes\nREASONING: Answering yes to Q3.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_default_yes.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="1",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Should have skipped Q2 and jumped to Q3
        assert "Dimension A" in dimension_answers
        assert "Dimension B" in dimension_answers
        assert len(evaluator.calls) == 2

    @pytest.mark.asyncio
    async def test_default_no_moves_to_next_question(self, fixtures_dir: Path):
        """Test answering 'No' without GOTO moves to next question."""
        from tests.mocks.mock_llm import MockLLM

        rubric_content = (
            "Question ID\tDimension\tSeverity\tRisk Type\t"
            "Question\tExamples\tAnswer\tGOTO\n"
            "1\tDimension A\tRed\t\tFirst question?\tTest example\tYes\t3\n"
            "\t\t\t\t\t\tNo\t\n"
            "2\tDimension A\tYellow\t\t"
            "Second question?\tTest example\tYes\tEND\n"
            "3\tDimension B\t\t\tThird question?\tTest example\tYes\tEND\n"
        )
        rubric_file = fixtures_dir / "rubric_default_no.tsv"
        rubric_file.write_text(rubric_content)

        mock_responses = [
            "ANSWER: No\nREASONING: Answering no to Q1.",
            "ANSWER: Yes\nREASONING: Answering yes to Q2.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_default_no.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="1",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Should have gone to Q2 instead of Q3
        assert "Dimension A" in dimension_answers
        assert len(dimension_answers["Dimension A"]) == 2
        assert dimension_answers["Dimension A"][0]["answer"] == "No"
        assert dimension_answers["Dimension A"][1]["answer"] == "Yes"

    @pytest.mark.asyncio
    async def test_infinite_loop_prevention(self, fixtures_dir: Path):
        """Test that visiting the same question twice stops evaluation."""
        from tests.mocks.mock_llm import MockLLM

        rubric_content = (
            "Question ID\tDimension\tSeverity\tRisk Type\t"
            "Question\tExamples\tAnswer\tGOTO\n"
            "1\tDimension A\t\t\tFirst question?\tTest example\tLoop\t1\n"
            "\t\t\t\t\t\tExit\tEND\n"
        )
        rubric_file = fixtures_dir / "rubric_circular.tsv"
        rubric_file.write_text(rubric_content)

        mock_responses = [
            "ANSWER: Loop\nREASONING: Creating circular reference.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_circular.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="1",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Should have stopped after first visit
        assert len(evaluator.calls) == 1

    @pytest.mark.asyncio
    async def test_question_not_found_stops_flow(self, fixtures_dir: Path):
        """Test that referencing non-existent question stops the flow."""
        from tests.mocks.mock_llm import MockLLM

        mock_responses = [
            "ANSWER: Yes\nREASONING: Answer.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_single_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        result = await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="999",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Should stop immediately
        assert result is None
        assert len(evaluator.calls) == 0

    @pytest.mark.asyncio
    async def test_severity_tracking_in_answers(self, fixtures_dir: Path):
        """Test that severity is tracked correctly in dimension answers."""
        from tests.mocks.mock_llm import MockLLM

        mock_responses = [
            "ANSWER: Yes\nREASONING: Red severity found.",
            "ANSWER: Yes\nREASONING: Yellow severity found.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_severity_values.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="1",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Verify severity was captured
        if "Dimension A" in dimension_answers:
            assert dimension_answers["Dimension A"][0]["severity"] == "Red"
        if "Dimension B" in dimension_answers:
            assert dimension_answers["Dimension B"][0]["severity"] == "Yellow"

    @pytest.mark.asyncio
    async def test_reasoning_extraction(self, fixtures_dir: Path):
        """Test that reasoning is extracted and stored correctly."""
        from tests.mocks.mock_llm import MockLLM

        mock_responses = [
            "ANSWER: Yes\nREASONING: This is the detailed reasoning for the answer.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_single_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="1",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Verify reasoning was extracted
        assert "Dimension A" in dimension_answers
        reasoning = dimension_answers["Dimension A"][0]["reasoning"]
        assert "detailed reasoning" in reasoning.lower()

    @pytest.mark.asyncio
    async def test_question_text_stored_in_answers(self, fixtures_dir: Path):
        """Test that question text is stored in dimension answers."""
        from tests.mocks.mock_llm import MockLLM

        mock_responses = [
            "ANSWER: Yes\nREASONING: Answer.",
        ]

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_single_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        evaluator = MockLLM(responses=mock_responses)
        evaluator.set_system_prompt("Test system prompt")
        dimension_answers = {}

        await judge._evaluate_question_flow(
            evaluator=evaluator,
            start_question_id="1",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Verify question text was stored
        assert "Dimension A" in dimension_answers
        answer_data = dimension_answers["Dimension A"][0]
        assert "question" in answer_data
        assert answer_data["question"] == "Single row question?"
        assert answer_data["question_id"] == "1"
