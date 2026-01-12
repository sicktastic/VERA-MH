"""Utilities for conversation management and file operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def generate_conversation_filename(prefix: str = "conversation") -> str:
    """
    Generate a timestamped filename for conversation logs.

    Args:
        prefix: Prefix for the filename

    Returns:
        Formatted filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.txt"


def save_conversation_to_file(
    conversation_history: List[Dict[str, Any]],
    filename: str,
    folder: str,
    llm1_name: Optional[str] = None,
) -> None:
    """
    Save conversation history to a text file.

    Args:
        conversation_history: List of conversation turns
        filename: Output filename
        folder: Output folder
        llm1_name: Name of LLM1 to identify it as 'user'
    """
    summary = format_conversation_summary(conversation_history, llm1_name)
    with open(f"{folder}/{filename}", "w", encoding="utf-8") as f:
        f.write(summary)


def format_conversation_summary(
    conversation_history: List[Dict[str, Any]], llm1_name: Optional[str] = None
) -> str:
    """
    Format conversation history into a readable summary.

    Args:
        conversation_history: List of conversation turns
        llm1_name: Name of LLM1 to identify it as 'user'

    Returns:
        Formatted conversation summary
    """
    if not conversation_history:
        return "No conversation recorded."

    summary = ""

    for i, turn in enumerate(conversation_history):
        speaker_name = turn["speaker"]

        # Determine if this is LLM1 (user) or LLM2 (agent)
        if llm1_name and llm1_name in speaker_name:
            prefix = "user:"
        else:
            prefix = "chatbot:"

        # Add the response with proper formatting
        summary += f"{prefix} {turn['response']}\n"

        # Check for early termination flag
        if turn.get("early_termination", False):
            summary += f"[CONVERSATION ENDED - {speaker_name} signaled termination]\n"

        # Add empty line between speakers, but not after the last turn
        if i < len(conversation_history) - 1:
            summary += "\n"

    return summary


def build_langchain_messages(
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    current_message: Optional[str] = None,
) -> List[BaseMessage]:
    """
    Build a list of LangChain messages from conversation history.

    Uses turn indices to determine message type since speaker names can be custom:
    - Odd turns (1, 3, 5...) are from persona (HumanMessage)
    - Even turns (2, 4, 6...) are from agent (AIMessage)

    IMPORTANT: This assumes persona always speaks first (see ConversationSimulator
    line 88-89). If the speaker order changes, this logic must be updated.

    Args:
        conversation_history: Optional list of previous conversation turns.
            Each turn is a dict with keys: 'turn', 'response', etc.
        current_message: Optional current message to add at the end

    Returns:
        List of LangChain message objects (HumanMessage, AIMessage)
    """
    messages = []

    # Add conversation history if provided
    if conversation_history:
        for turn in conversation_history:
            turn_number = turn.get("turn")
            text = turn.get("response")
            # Skip turns without turn number
            if turn_number is None:
                continue
            # Odd turns (1, 3, 5...) are from persona (HumanMessage)
            # Even turns (2, 4, 6...) are from agent (AIMessage)
            if turn_number % 2 == 1:
                messages.append(HumanMessage(content=text))
            else:
                messages.append(AIMessage(content=text))

    # Add current message
    if current_message:
        messages.append(HumanMessage(content=current_message))

    return messages
