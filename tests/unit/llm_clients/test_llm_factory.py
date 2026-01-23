from unittest.mock import patch

import pytest

from llm_clients.azure_llm import AzureLLM
from llm_clients.claude_llm import ClaudeLLM
from llm_clients.gemini_llm import GeminiLLM
from llm_clients.llama_llm import LlamaLLM
from llm_clients.llm_factory import LLMFactory
from llm_clients.openai_llm import OpenAILLM


@pytest.mark.unit
class TestLLMFactory:
    """Unit tests for LLMFactory class."""

    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    def test_create_claude_llm(self):
        """Test that factory correctly creates Claude LLM instance."""
        # Arrange
        model_name = "claude-3-5-sonnet-20241022"
        name = "TestClaude"
        system_prompt = "You are a helpful assistant."

        # Act
        llm = LLMFactory.create_llm(
            model_name=model_name, name=name, system_prompt=system_prompt
        )

        # Assert
        assert isinstance(llm, ClaudeLLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == model_name

    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    def test_create_openai_llm(self):
        """Test that factory correctly creates OpenAI LLM instance."""
        # Arrange
        model_name = "gpt-4"
        name = "TestGPT"
        system_prompt = "You are a test assistant."

        # Act
        llm = LLMFactory.create_llm(
            model_name=model_name, name=name, system_prompt=system_prompt
        )

        # Assert
        assert isinstance(llm, OpenAILLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == model_name

    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    def test_create_gemini_llm(self):
        """Test that factory correctly creates Gemini LLM instance."""
        # Arrange
        model_name = "gemini-pro"
        name = "TestGemini"
        system_prompt = "You are a Gemini assistant."

        # Act
        llm = LLMFactory.create_llm(
            model_name=model_name, name=name, system_prompt=system_prompt
        )

        # Assert
        assert isinstance(llm, GeminiLLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == model_name

    def test_create_llama_llm(self):
        """Test that factory correctly creates Llama LLM instance."""
        # Arrange
        model_name = "llama-3.1"
        name = "TestLlama"
        system_prompt = "You are a Llama assistant."

        # Act
        llm = LLMFactory.create_llm(
            model_name=model_name, name=name, system_prompt=system_prompt
        )

        # Assert
        assert isinstance(llm, LlamaLLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == model_name

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    def test_create_azure_llm(self):
        """Test that factory correctly creates Azure LLM instance."""
        # Arrange
        model_name = "azure-grok-4"
        name = "TestAzure"
        system_prompt = "You are an Azure assistant."

        with patch(
            "llm_clients.azure_llm.Config.get_azure_config",
            return_value={"model": "azure-gpt-4"},
        ):
            # Act
            llm = LLMFactory.create_llm(
                model_name=model_name, name=name, system_prompt=system_prompt
            )

            # Assert
            assert isinstance(llm, AzureLLM)
            assert llm.name == name
            assert llm.system_prompt == system_prompt
            assert llm.model_name == "grok-4"  # azure- prefix should be removed

    def test_unsupported_model_raises_error(self):
        """Test that factory raises ValueError for unsupported model names."""
        # Arrange
        unsupported_model = "unknown-model-xyz"
        name = "TestUnsupported"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create_llm(model_name=unsupported_model, name=name)

        assert "Unsupported model" in str(exc_info.value)
        assert unsupported_model in str(exc_info.value)

    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    def test_factory_passes_kwargs(self, mock_chat_anthropic):
        """Test that factory correctly forwards kwargs to LLM implementations."""
        # Arrange
        model_name = "claude-3-5-sonnet-20241022"
        name = "TestKwargs"
        temperature = 0.5
        max_tokens = 500

        # Act
        llm = LLMFactory.create_llm(
            model_name=model_name,
            name=name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Assert
        assert isinstance(llm, ClaudeLLM)
        # Verify kwargs were passed to underlying LangChain model
        mock_chat_anthropic.assert_called_once()
        call_kwargs = mock_chat_anthropic.call_args[1]
        assert call_kwargs["temperature"] == temperature
        assert call_kwargs["max_tokens"] == max_tokens

    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    def test_factory_filters_non_model_params(self, mock_chat_openai):
        """Test that factory filters out non-model-specific parameters."""
        # Arrange
        model_name = "gpt-4"
        name = "TestFiltering"
        temperature = 0.7
        # These should be filtered out (model, name, prompt_name, system_prompt)
        extra_params = {
            "prompt_name": "should-be-ignored",
        }

        # Act
        llm = LLMFactory.create_llm(
            model_name=model_name,
            name=name,
            temperature=temperature,
            **extra_params,
        )

        # Assert
        assert isinstance(llm, OpenAILLM)
        assert llm.name == name
        assert llm.model_name == model_name
        # Verify that filtered params were not passed to ChatOpenAI
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["temperature"] == temperature
        assert "prompt_name" not in call_kwargs  # Filtered param should not be present

    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    def test_create_openai_llm_with_openai_prefix(self, mock_chat_openai):
        """Test that factory correctly identifies OpenAI models with 'openai' prefix."""
        # Arrange
        model_name = "openai-custom-model"
        name = "TestOpenAIPrefix"

        # Act
        llm = LLMFactory.create_llm(model_name=model_name, name=name)

        # Assert
        assert isinstance(llm, OpenAILLM)
        assert llm.model_name == model_name

    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    def test_create_gemini_llm_with_google_prefix(self):
        """Test that factory correctly identifies Gemini models with 'google' prefix."""
        # Arrange
        model_name = "google-gemini-ultra"
        name = "TestGooglePrefix"

        # Act
        llm = LLMFactory.create_llm(model_name=model_name, name=name)

        # Assert
        assert isinstance(llm, GeminiLLM)
        assert llm.model_name == model_name

    def test_create_llama_llm_with_ollama_prefix(self):
        """Test that factory correctly identifies Llama models with 'ollama' prefix."""
        # Arrange
        model_name = "ollama-llama-3"
        name = "TestOllamaPrefix"

        # Act
        llm = LLMFactory.create_llm(model_name=model_name, name=name)

        # Assert
        assert isinstance(llm, LlamaLLM)
        assert llm.model_name == model_name

    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    def test_factory_case_insensitive_model_detection(self):
        """Test that factory detects models regardless of case."""
        # Arrange
        with patch(
            "llm_clients.azure_llm.Config.get_azure_config",
            return_value={"model": "azure-gpt-4"},
        ):
            # Act
            claude_llm = LLMFactory.create_llm(model_name="CLAUDE-3-5", name="Claude")
            gpt_llm = LLMFactory.create_llm(model_name="GPT-4-TURBO", name="GPT")
            gemini_llm = LLMFactory.create_llm(model_name="GEMINI-PRO", name="Gemini")
            llama_llm = LLMFactory.create_llm(model_name="LLAMA-3", name="Llama")
            azure_llm = LLMFactory.create_llm(model_name="AZURE-GROK-4", name="Azure")

            # Assert
            assert isinstance(claude_llm, ClaudeLLM)
            assert isinstance(gpt_llm, OpenAILLM)
            assert isinstance(gemini_llm, GeminiLLM)
            assert isinstance(llama_llm, LlamaLLM)
            assert isinstance(azure_llm, AzureLLM)
