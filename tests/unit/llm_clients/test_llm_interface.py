import uuid
from typing import Optional
from unittest.mock import MagicMock

import pytest

from llm_clients import Role
from llm_clients.llm_interface import LLMInterface


class ConcreteLLM(LLMInterface):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, name: str, role: Role, system_prompt: Optional[str] = None):
        super().__init__(name, role, system_prompt)
        # Add a mock llm object for __getattr__ testing
        self.llm = MagicMock(spec=["temperature", "max_tokens", "custom_method"])
        self.llm.temperature = 0.7
        self.llm.max_tokens = 1000

    async def start_conversation(self) -> str:
        """Concrete implementation of abstract method."""
        return "test response"

    async def generate_response(self, conversation_history=None):
        """Concrete implementation of abstract method."""
        return "test response"

    def set_system_prompt(self, system_prompt: str) -> None:
        """Concrete implementation of abstract method."""
        self.system_prompt = system_prompt


class IncompleteLLM(LLMInterface):
    """Incomplete implementation to test abstract method enforcement."""

    pass


@pytest.mark.unit
class TestLLMInterface:
    """Unit tests for LLMInterface abstract base class."""

    def test_init_with_name_only(self):
        """Test initialization with only name parameter."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)

        assert llm.name == "TestLLM"
        assert llm.system_prompt == ""

    def test_init_with_name_and_system_prompt(self):
        """Test initialization with name and system prompt."""
        prompt = "You are a helpful assistant."
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER, system_prompt=prompt)

        assert llm.name == "TestLLM"
        assert llm.system_prompt == prompt

    @pytest.mark.asyncio
    async def test_generate_response_abstract_method(self, mock_system_message):
        """Test that generate_response is implemented in concrete class (line 21)."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "test response"

    def test_set_system_prompt_abstract_method(self):
        """Test that set_system_prompt is implemented in concrete class (line 26)."""
        llm = ConcreteLLM(
            name="TestLLM", role=Role.PROVIDER, system_prompt="Initial prompt"
        )
        assert llm.system_prompt == "Initial prompt"

        llm.set_system_prompt("Updated prompt")
        assert llm.system_prompt == "Updated prompt"

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMInterface cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            LLMInterface(name="Test", role=Role.PROVIDER)  # pyright: ignore[reportAbstractUsage]

        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_incomplete_implementation_raises_error(self):
        """Test that incomplete implementations raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            IncompleteLLM(name="Incomplete", role=Role.PROVIDER)  # pyright: ignore[reportAbstractUsage]

        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_getattr_delegates_to_llm(self):
        """Test that __getattr__ delegates to self.llm (lines 40-41)."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)

        # Access attributes that exist on llm
        assert llm.temperature == 0.7
        assert llm.max_tokens == 1000

    def test_getattr_raises_attribute_error_for_missing_attribute(self):
        """Test that __getattr__ raises AttributeError for missing attributes."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)

        # Try to access attribute that doesn't exist on llm (spec prevents it)
        with pytest.raises(AttributeError) as exc_info:
            _ = llm.nonexistent_attribute

        assert "ConcreteLLM" in str(exc_info.value)
        assert "nonexistent_attribute" in str(exc_info.value)

    def test_getattr_when_llm_not_set(self):
        """Test __getattr__ behavior when self.llm doesn't exist."""

        class MinimalLLM(LLMInterface):
            """Minimal implementation without self.llm."""

            async def start_conversation(self) -> str:
                return "response"

            async def generate_response(self, conversation_history=None):
                return "response"

            def set_system_prompt(self, system_prompt: str) -> None:
                self.system_prompt = system_prompt

        llm = MinimalLLM(name="Minimal", role=Role.PROVIDER)

        # Should raise AttributeError
        with pytest.raises(AttributeError):
            _ = llm.some_attribute

    def test_getattr_with_none_llm(self):
        """Test __getattr__ when self.llm is None."""

        class NullLLM(LLMInterface):
            """Implementation with None llm."""

            def __init__(
                self, name: str, role: Role, system_prompt: Optional[str] = None
            ):
                super().__init__(name, role, system_prompt)
                self.llm = None

            async def start_conversation(self) -> str:
                return "response"

            async def generate_response(self, conversation_history=None):
                return "response"

            def set_system_prompt(self, system_prompt: str) -> None:
                self.system_prompt = system_prompt

        llm = NullLLM(name="Null", role=Role.PROVIDER)

        # Should raise AttributeError since llm is None
        with pytest.raises(AttributeError):
            _ = llm.temperature

    def test_multiple_instances_have_independent_state(self):
        """Test that multiple LLM instances maintain independent state."""
        llm1 = ConcreteLLM(name="LLM1", role=Role.PROVIDER, system_prompt="Prompt 1")
        llm2 = ConcreteLLM(name="LLM2", role=Role.PROVIDER, system_prompt="Prompt 2")

        assert llm1.name == "LLM1"
        assert llm2.name == "LLM2"
        assert llm1.system_prompt == "Prompt 1"
        assert llm2.system_prompt == "Prompt 2"
        assert llm1.conversation_id != llm2.conversation_id

        # Modify one shouldn't affect the other
        llm1.set_system_prompt("Modified Prompt 1")
        assert llm1.system_prompt == "Modified Prompt 1"
        assert llm2.system_prompt == "Prompt 2"

    def test_getattr_with_callable_attribute(self):
        """Test __getattr__ works with callable attributes."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        llm.llm.custom_method = MagicMock(return_value="method result")

        # Access callable attribute through delegation
        result = llm.custom_method()
        assert result == "method result"
        llm.llm.custom_method.assert_called_once()

    def test_system_prompt_default_empty_string(self):
        """Test that system_prompt defaults to empty string, not None."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        assert llm.system_prompt == ""
        assert llm.system_prompt is not None

    def test_getattr_preserves_attribute_type(self):
        """Test that __getattr__ preserves the type of delegated attributes."""

        # Create a fresh mock without spec for this test
        class FlexibleLLM(LLMInterface):
            def __init__(
                self, name: str, role: Role, system_prompt: Optional[str] = None
            ):
                super().__init__(name, role, system_prompt)
                self.llm = MagicMock()
                self.llm.string_attr = "test string"
                self.llm.int_attr = 42
                self.llm.float_attr = 3.14
                self.llm.bool_attr = True
                self.llm.list_attr = [1, 2, 3]

            async def start_conversation(self) -> str:
                return "response"

            async def generate_response(self, conversation_history=None):
                return "response"

            def set_system_prompt(self, system_prompt: str) -> None:
                self.system_prompt = system_prompt

        llm = FlexibleLLM(name="TestLLM", role=Role.PROVIDER)

        assert isinstance(llm.string_attr, str)
        assert isinstance(llm.int_attr, int)
        assert isinstance(llm.float_attr, float)
        assert isinstance(llm.bool_attr, bool)
        assert isinstance(llm.list_attr, list)

    def test_init_sets_conversation_id(self):
        """Test that conversation_id is set at init (e.g. UUID)."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        assert llm.conversation_id is not None
        assert isinstance(llm.conversation_id, str)
        assert len(llm.conversation_id) > 0

    def test_create_conversation_id_returns_string(self):
        """Test that create_conversation_id returns a non-empty string (e.g. UUID)."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        cid = llm.create_conversation_id()
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_create_conversation_id_returns_distinct_valid_uuid(self):
        """Test that repeated calls return distinct values and each is a valid UUID."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        ids = [llm.create_conversation_id() for _ in range(50)]
        assert len(ids) == len(set(ids)), "ids must be distinct"
        for cid in ids:
            uuid.UUID(cid)  # valid UUID string

    def test_update_conversation_id_from_metadata_leaves_unchanged_when_absent(self):
        """
        Test _update_conversation_id_from_metadata preserves conversation_id
        when key is absent from response metadata.
        """
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        original = llm.conversation_id
        llm._last_response_metadata = {}
        llm._update_conversation_id_from_metadata()
        assert llm.conversation_id == original

    def test_update_conversation_id_from_metadata_overwrites_when_present(self):
        """
        Test _update_conversation_id_from_metadata overwrites self.conversation_id
        with API-returned conversation_id.
        """
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        llm._last_response_metadata = {"conversation_id": "api-provided-id"}
        llm._update_conversation_id_from_metadata()
        assert llm.conversation_id == "api-provided-id"

    @pytest.mark.asyncio
    async def test_conversation_id_available_after_generate_response(self):
        """Test that conversation_id remains set after start_conversation."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        await llm.start_conversation()
        assert llm.conversation_id is not None


@pytest.mark.unit
class TestLLMInterfaceHooks:
    """Tests for the extensibility hooks added to LLMInterface."""

    def test_bespoke_termination_signals_default_empty(self):
        """bespoke_termination_signals returns [] by default."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        assert llm.bespoke_termination_signals == []

    def test_post_process_response_noop(self):
        """_post_process_response is a no-op by default."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        assert llm._post_process_response("hello world") == "hello world"
        assert llm._post_process_response("") == ""

    def test_should_discard_response_default_false(self):
        """_should_discard_response returns False regardless of signals."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        assert llm._should_discard_response([]) is False
        assert llm._should_discard_response(["SIGNAL"]) is False

    def test_extract_signals_returns_matching_signals(self):
        """_extract_signals returns only the signals found in the response."""

        class SignalLLM(ConcreteLLM):
            @property
            def bespoke_termination_signals(self):
                return ["[END]", "[ERROR]"]

        llm = SignalLLM("test", Role.PROVIDER)
        found = llm._extract_signals("Something happened [END] here")
        assert found == ["[END]"]

    def test_extract_signals_returns_empty_when_no_match(self):
        """_extract_signals returns [] when no signals are present."""

        class SignalLLM(ConcreteLLM):
            @property
            def bespoke_termination_signals(self):
                return ["[END]"]

        llm = SignalLLM("test", Role.PROVIDER)
        assert llm._extract_signals("nothing special here") == []

    def test_extract_signals_returns_multiple_matches(self):
        """_extract_signals returns all signals found in the response."""

        class SignalLLM(ConcreteLLM):
            @property
            def bespoke_termination_signals(self):
                return ["[END]", "[ERROR]"]

        llm = SignalLLM("test", Role.PROVIDER)
        found = llm._extract_signals("[ERROR] something went wrong [END]")
        assert "[END]" in found
        assert "[ERROR]" in found

    def test_extract_signals_case_insensitive(self):
        """_extract_signals is case-insensitive."""

        class SignalLLM(ConcreteLLM):
            @property
            def bespoke_termination_signals(self):
                return ["[end]"]

        llm = SignalLLM("test", Role.PROVIDER)
        assert llm._extract_signals("message [END] here") == ["[end]"]

    def test_first_speaker_default_none(self):
        """first_speaker returns None by default."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        assert llm.first_speaker is None

    def test_prepare_sessions_returns_unchanged(self):
        """prepare_sessions returns the same list unmodified."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        sessions = ["intake", "coaching"]
        result = llm.prepare_sessions(sessions)
        assert result == sessions

    @pytest.mark.asyncio
    async def test_setup_is_noop(self):
        """setup() completes without error by default."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        await llm.setup()  # must not raise

    @pytest.mark.asyncio
    async def test_finish_and_reset_session_is_noop(self):
        """finish_and_reset_session() completes without error by default."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        await llm.finish_and_reset_session("intake")  # must not raise
