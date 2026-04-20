#!/usr/bin/env python3

import asyncio
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from llm_clients import LLMFactory
from llm_clients.llm_interface import Role
from utils.logging_utils import (
    cleanup_logger,
    log_conversation_end,
    log_conversation_start,
    log_conversation_turn,
    setup_conversation_logger,
)

from .conversation_simulator import ConversationSimulator
from .utils import load_prompts_from_csv


class ConversationRunner:
    """Handles running LLM conversations with logging and file management."""

    def __init__(
        self,
        persona_model_config: Dict[str, Any],
        agent_model_config: Dict[str, Any],
        run_id: str,
        max_turns: int = 6,
        runs_per_prompt: int = 3,
        folder_name: str = "conversations",
        max_concurrent: Optional[int] = None,
        max_total_words: Optional[int] = None,
        max_personas: Optional[int] = None,
        persona_speaks_first: Optional[bool] = None,
        session_types: Optional[List[str]] = None,
    ):
        self.persona_model_config = persona_model_config
        self.agent_model_config = agent_model_config
        self.max_turns = max_turns
        self.runs_per_prompt = runs_per_prompt
        self.folder_name = folder_name
        self.run_id = run_id

        # Limit concurrent conversations to avoid overwhelming the server
        # Default: None - run all conversations concurrently
        self.max_concurrent = max_concurrent
        self.max_total_words = max_total_words
        self.max_personas = max_personas
        self.persona_speaks_first = persona_speaks_first
        self.session_types = session_types

        # Default persona_speaks_first based on agent type
        if persona_speaks_first is None:
            agent_model = agent_model_config.get("model", "").lower()
        else:
            self.persona_speaks_first = persona_speaks_first


    async def run_single_conversation(
        self,
        persona_config: dict,
        max_turns: int,
        conversation_index: int,
        run_number: int,
        **kwargs: dict,
    ) -> Dict[str, Any]:
        """Run a simulated conversation (persona vs provider LLM), across one or more sessions.

        A single session without --sessions is treated as a one-element session list so
        the same code path handles both cases. The agent's prepare_sessions() hook
        normalises the list (e.g. prepending INTAKE for ray-backend).

        Args:
            persona_config (dict): Must have "model", "prompt", "name".
            max_turns (int): Max conversation turns per session.
            conversation_index (int): Index in the batch of conversations.
            run_number (int): Run index for this prompt (e.g. 1 of runs_per_prompt).
            **kwargs: Unused; reserved for future use.

        Returns:
            Dict[str, Any]: index, llm1_model, llm1_prompt, run_number, turns,
            filenames, log_files, duration, early_termination, conversation.
        """
        model_name = persona_config["model"]
        system_prompt = persona_config["prompt"]  # This is now the full persona prompt
        persona_name = persona_config["name"]

        # Generate filename base using persona name, model, and run number
        tag = uuid.uuid4().hex[:6]
        # TODO: should this be inside the LLM class?
        model_short = (
            model_name.replace("claude-3-", "c3-")
            .replace("gpt-", "g")
            .replace("claude-sonnet-4-", "cs4-")
        )
        persona_safe = persona_name.replace(" ", "_").replace(".", "")
        workflow = self.agent_model_config.get("workflow", "")
        workflow_part = f"_{workflow}" if workflow else ""
        filename_base = f"{tag}_{persona_safe}_{model_short}{workflow_part}_run{run_number}"
        os.makedirs(f"{self.folder_name}", exist_ok=True)

        # Setup top-level logger (conversation start/end metadata)
        logger = setup_conversation_logger(filename_base, run_id=self.run_id)
        start_time = time.time()

        # Create persona instance
        persona = LLMFactory.create_llm(
            model_name=model_name,
            name=f"{model_short} {persona_name}",
            system_prompt=system_prompt,
            role=Role.PERSONA,
            **self.persona_model_config,
        )

        # Create new agent instance to reset conversation_id and metadata.
        # Exclude selected kwargs to avoid duplicate args expected in create_llm.
        agent_kwargs = {
            k: v
            for k, v in self.agent_model_config.items()
            if k not in ("model", "name", "system_prompt")
        }
        agent_kwargs["user_name"] = persona_name

        agent = LLMFactory.create_llm(
            model_name=self.agent_model_config["model"],
            name=self.agent_model_config.get("name", "Provider"),
            system_prompt=self.agent_model_config.get(
                "system_prompt", "You are a helpful AI assistant."
            ),
            role=Role.PROVIDER,
            **agent_kwargs,
        )

        # Log conversation start
        log_conversation_start(
            logger=logger,
            llm1_model_str=model_name,
            llm1_prompt=persona_name,
            llm2_name=agent.name,
            llm2_model_str=getattr(agent, "model_name", "unknown"),
            max_turns=max_turns,
            persona_speaks_first=self.persona_speaks_first,
            llm1_model=persona,
            llm2_model=agent,
        )

        all_conversations: List[Dict[str, Any]] = []
        filenames: List[str] = []
        log_files: List[str] = []

        try:
            await agent.setup()

            # Always work with a session list; agent normalises it (e.g. prepends INTAKE)
            raw_sessions = self.session_types or [getattr(agent, "_session_type", "default")]
            session_types = agent.prepare_sessions(raw_sessions)

            for i, session_type in enumerate(session_types, 1):
                if i > 1:
                    print(f"  Session {i - 1} finished. Starting session {i}/{len(session_types)}: {session_type}")
                else:
                    print(f"  Starting session {i}/{len(session_types)}: {session_type}")

                await agent.finish_and_reset_session(session_type)

                first_speaker = agent.first_speaker
                # psp = persona speaks first
                psf = (first_speaker == Role.PERSONA) if first_speaker is not None else self.persona_speaks_first

                # Use session-type suffix in filename only when --sessions was explicitly passed
                session_filename_base = (
                    f"{filename_base}_{i}_{session_type}" if self.session_types else filename_base
                )
                session_logger = setup_conversation_logger(session_filename_base, run_id=self.run_id)
                session_start = time.time() 

                simulator = ConversationSimulator(persona, agent)
                conversation = await simulator.generate_conversation(
                    max_turns=max_turns,
                    max_total_words=self.max_total_words,
                    persona_speaks_first=psf,
                )
                self._log_turns(session_logger, conversation)
                session_time = time.time() - session_start
                session_early_term = any(t.get("early_termination", False) for t in conversation)
                log_conversation_end(session_logger, len(conversation), session_early_term, session_time)

                output_filename = f"{session_filename_base}.txt"
                simulator.save_conversation(output_filename, self.folder_name)
                all_conversations.extend(conversation)
                filenames.append(f"{self.folder_name}/{output_filename}")
                log_files.append(f"logging/{self.run_id}/{session_filename_base}.log")
                cleanup_logger(session_logger)

        finally:
            cleanup_logger(logger)
            for llm in (persona, agent):
                try:
                    await llm.cleanup()
                except Exception as e:
                    print(f"Warning: Failed to cleanup LLM: {e}")

        end_time = time.time()
        early_termination = any(t.get("early_termination", False) for t in all_conversations)

        return {
            "index": conversation_index,
            "llm1_model": model_name,
            "llm1_prompt": persona_name,
            "run_number": run_number,
            "turns": len(all_conversations),
            "filenames": filenames,
            "log_files": log_files,
            "duration": end_time - start_time,
            "early_termination": early_termination,
            "conversation": all_conversations,
        }

    def _log_turns(self, logger, conversation: List[Dict[str, Any]]) -> None:
        for i, turn in enumerate(conversation, 1):
            log_conversation_turn(
                logger=logger,
                turn_number=i,
                speaker=turn.get("speaker", "Unknown"),
                input_message=turn.get("input", ""),
                response=turn.get("response", ""),
                early_termination=turn.get("early_termination", False),
                logging=turn.get("logging", {}),
            )

    async def run_conversations(
        self, persona_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Run multiple conversations concurrently."""
        # Load prompts from CSV based on persona names
        personas = load_prompts_from_csv(persona_names, max_personas=self.max_personas)

        # Create tasks for all conversations (each prompt run multiple times)
        tasks = []
        conversation_index = 1

        for persona in personas:
            for run in range(1, self.runs_per_prompt + 1):
                tasks.append(
                    self.run_single_conversation(
                        {
                            "model": self.persona_model_config["model"],
                            "prompt": persona["prompt"],
                            "name": persona["Name"],
                            "run": run,
                        },
                        self.max_turns,
                        conversation_index,
                        run,
                    )
                )
                conversation_index += 1

        # Run all conversations with concurrency limit
        start_time = datetime.now()

        if self.max_concurrent and len(tasks) > self.max_concurrent:
            # Use semaphore to limit concurrent conversations
            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def run_with_limit(task):
                async with semaphore:
                    return await task

            print(
                f"Running {len(tasks)} conversations with max concurrency: "
                f"{self.max_concurrent}"
            )
            results = await asyncio.gather(*[run_with_limit(task) for task in tasks])
        else:
            # Run all conversations concurrently (no limit)
            print(f"Running {len(tasks)} conversations concurrently (no limit)")
            results = await asyncio.gather(*tasks)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        print(f"\nCompleted {len(results)} conversations in {total_time:.2f} seconds")

        return results
