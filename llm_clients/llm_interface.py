import copy
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Role(Enum):
    """Role of the LLM in a conversation."""

    PERSONA = "persona"  # User roleplaying as a human seeking help
    PROVIDER = "provider"  # Chatbot providing support
    JUDGE = "judge"  # Judge role, used for judge operations


class LLMInterface(ABC):
    """Abstract base class for LLM implementations.

    Provides basic text generation capabilities. All LLM implementations
    must support basic text generation and system prompt management.
    """

    def __init__(
        self,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt or ""
        self._last_response_metadata: Dict[str, Any] = {}
        self.conversation_id: Optional[str] = None

    @property
    def last_response_metadata(self) -> Dict[str, Any]:
        """Metadata from the last generate_response call. Returns a deep copy so
        callers cannot mutate internal state (including nested dicts like usage).
        """
        return copy.deepcopy(self._last_response_metadata)

    @last_response_metadata.setter
    def last_response_metadata(self, value: Optional[Dict[str, Any]]) -> None:
        """Set metadata; use _last_response_metadata for in-place updates."""
        self._last_response_metadata = value or {}

    def create_conversation_id(self) -> str:
        """Create a new unique conversation id.

        Used when the client does not provide one in response metadata.
        Subclasses may override to use a different id format.
        """
        return str(uuid.uuid4())

    def ensure_conversation_id(self) -> None:
        """Set conversation_id from last response metadata or create one if not set.

        Call after generate_response (e.g. at the end of generate_response).
        If conversation_id is already set, does nothing. Otherwise sets it from
        self.last_response_metadata["conversation_id"] if present, or from
        create_conversation_id(). Implementations must update
        self.last_response_metadata in generate_response().
        """
        if self.conversation_id is not None:
            return
        metadata = self.last_response_metadata or {}
        self.conversation_id = (
            metadata.get("conversation_id") or self.create_conversation_id()
        )

    @abstractmethod
    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a response based on conversation history.

        Args:
            conversation_history: List of previous conversation turns.
                Each turn is a dict with keys: 'turn', 'speaker', 'response'.
                On the first turn (turn 0), conversation_history will contain
                a single entry with turn=0, speaker="system", and the initial
                message in the 'response' field. This provides context for
                starting the conversation.

        Returns:
            str: The response text. Metadata in self.last_response_metadata
                (getter returns a copy so callers need not copy).

        Note:
            For API thread/session identification, use self.conversation_id.
            Implementations should call self.ensure_conversation_id() before
            returning, so self.conversation_id is set for the next call.
        """
        pass

    @abstractmethod
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        pass

    async def cleanup(self) -> None:
        """Clean up any resources used by this LLM instance.

        Subclasses that need cleanup (like AzureLLM) should override this method.
        Default implementation does nothing.
        """
        pass

    def __getattr__(self, name):
        """Delegate attribute access to the underlying llm object.

        This allows accessing attributes like temperature, max_tokens, etc.
        directly on the LLM instance, which will be forwarded to the
        underlying LangChain model (self.llm).
        """
        # Check if self.llm exists by looking in __dict__ to avoid recursion
        # Only delegate if self.llm exists and has the attribute
        if "llm" in self.__dict__ and hasattr(self.llm, name):
            return getattr(self.llm, name)
        # If the attribute doesn't exist on self.llm, raise AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


class JudgeLLM(LLMInterface):
    """Extended LLM interface that supports structured output generation.

    This interface is required for LLM implementations that can be used
    as judges, where structured output (using Pydantic models) is necessary
    for reliable evaluation results.

    Implementations: Claude, OpenAI, Gemini, Azure
    Not supported by: Ollama
    """

    @abstractmethod
    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        """Generate a structured response using Pydantic model.

        Args:
            message: The prompt message
            response_model: Pydantic model class to structure the response

        Returns:
            Instance of the response_model with structured data

        Raises:
            RuntimeError: If structured output generation fails
        """
        pass
