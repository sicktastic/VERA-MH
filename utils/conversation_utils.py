"""Utilities for conversation management and file operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .debug import debug_print


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


def build_persona_role_reminder() -> str:
    """
    Build a role reminder message for personas to prevent role confusion.

    This reminder is injected before conversation history to reinforce that
    the persona should maintain their character as a user in distress, not
    adopt the chatbot's supportive counselor role.

    Returns:
        The role reminder text to inject as a HumanMessage
    """
    return (
        "REMINDER: You are the USER seeking help from the chatbot. "
        "The conversation below shows YOUR messages (what you said) and "
        "the CHATBOT's responses (what it said). Continue responding AS "
        "YOURSELF - the user in distress. Do NOT provide advice, support, "
        "or counseling. Do NOT copy the chatbot's supportive language."
    )


def build_langchain_messages(
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    current_message: Optional[str] = None,
    system_prompt: Optional[str] = None,
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
        current_message: Optional current message to add at the end.
            NOTE: If current_message matches the last message in history, it will
            NOT be added again to avoid duplication.
        system_prompt: Optional system prompt to check if this is a persona.
            If provided and contains "roleplaying as a human user", a role
            reminder will be automatically injected before conversation history.

    Returns:
        List of LangChain message objects (HumanMessage, AIMessage)
    """
    messages = []

    # Auto-detect persona and add role reminder if needed
    is_persona = system_prompt and "roleplaying as a human user" in system_prompt
    if is_persona and conversation_history:
        debug_print("[DEBUG] Adding role reminder message for persona")
        messages.append(HumanMessage(content=build_persona_role_reminder()))

    # Add conversation history if provided
    if conversation_history:
        hist_len = len(conversation_history)
        debug_print(f"[DEBUG] Processing {hist_len} turns from history:")
        for turn in conversation_history:
            turn_number = turn.get("turn")
            text = turn.get("response")
            # Skip turns without turn number or response
            if turn_number is None or text is None:
                continue
            # Odd turns (1, 3, 5...) are from persona (HumanMessage)
            # Even turns (2, 4, 6...) are from agent (AIMessage)
            msg_type = "HumanMessage" if turn_number % 2 == 1 else "AIMessage"
            preview = text[:50] + "..." if len(text) > 50 else text
            debug_print(f"  Turn {turn_number} -> {msg_type}: {preview}")
            if turn_number % 2 == 1:
                messages.append(HumanMessage(content=text))
            else:
                messages.append(AIMessage(content=text))

    # Add current message only if it's not already the last message in history
    # This prevents duplication when current_message is the same as the last response
    if current_message:
        # Check if we already added this message from history
        if not messages or messages[-1].content != current_message:
            debug_print(
                "[DEBUG build_langchain_messages] Adding current_message as "
                "HumanMessage (not duplicate)"
            )
            messages.append(HumanMessage(content=current_message))
        else:
            debug_print(
                "[DEBUG build_langchain_messages] Skipping current_message "
                "(duplicate of last message)"
            )

    return messages


def format_conversation_as_string(
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    current_message: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Format conversation history as a string for string-based LLMs (e.g., Ollama).

    This function reuses build_langchain_messages() and converts the result to
    a string format with Human/Assistant labels.

    Args:
        conversation_history: Optional list of previous conversation turns
        current_message: Optional current message to add at the end
        system_prompt: Optional system prompt to prepend

    Returns:
        Formatted string with System, Human, and Assistant labels
    """
    full_message = ""

    # Add system prompt if provided
    if system_prompt:
        full_message = f"System: {system_prompt}\n\n"

    # Build LangChain messages using existing utility
    messages = build_langchain_messages(conversation_history, current_message)

    # Convert messages to string format
    for message in messages:
        if isinstance(message, HumanMessage):
            full_message += f"Human: {message.content}\n\n"
        elif isinstance(message, AIMessage):
            full_message += f"Assistant: {message.content}\n\n"

    # Add "Assistant:" prompt at the end if there's a current message
    if current_message and full_message.endswith("\n\n"):
        full_message += "Assistant:"

    return full_message
