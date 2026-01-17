from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from llm_clients.azure_llm import AzureLLM


# Helper class for mocking response_metadata that supports both dict and
# attribute access
class DictWithAttr(dict):
    """Dict that supports both dict operations and attribute access."""

    def __getattr__(self, key):
        return self.get(key)


@pytest.mark.unit
class TestAzureLLM:
    """Unit tests for AzureLLM class."""

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", None)
    def test_init_missing_api_key_raises_error(self):
        """Test that missing AZURE_API_KEY raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AzureLLM(name="TestAzure")

        assert "AZURE_API_KEY not found" in str(exc_info.value)

    @patch("llm_clients.azure_llm.Config.AZURE_ENDPOINT", None)
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    def test_init_missing_endpoint_raises_error(self):
        """Test that missing AZURE_ENDPOINT raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AzureLLM(name="TestAzure")

        assert "AZURE_ENDPOINT not found" in str(exc_info.value)

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    def test_init_with_default_model(self, mock_azure_model, mock_get_config):
        """Test initialization with default model from config."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", system_prompt="Test prompt")

        assert llm.name == "TestAzure"
        assert llm.system_prompt == "Test prompt"
        assert llm.model_name == "gpt-4"
        assert llm.last_response_metadata == {}

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    def test_init_with_custom_model(self, mock_azure_model, mock_get_config):
        """Test initialization with custom model name."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4-turbo"
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", model_name="azure-gpt-4-turbo")

        assert llm.model_name == "gpt-4-turbo"  # azure- prefix should be removed

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    def test_init_removes_azure_prefix(self, mock_azure_model, mock_get_config):
        """Test that azure- prefix is removed from model name."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "grok-4"
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", model_name="azure-grok-4")

        assert llm.model_name == "grok-4"

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    def test_init_with_kwargs(self, mock_azure_model, mock_get_config):
        """Test initialization with additional kwargs."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_azure_model.return_value = mock_llm

        AzureLLM(name="TestAzure", temperature=0.5, max_tokens=500, top_p=0.9)

        # Verify kwargs were passed to AzureAIChatCompletionsModel
        call_kwargs = mock_azure_model.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.9

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.AZURE_API_VERSION", "2024-05-01-preview")
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    def test_init_with_api_version(self, mock_azure_model, mock_get_config):
        """Test initialization with API version from config."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")

        assert llm.api_version == "2024-05-01-preview"
        call_kwargs = mock_azure_model.call_args[1]
        assert call_kwargs["api_version"] == "2024-05-01-preview"

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.AZURE_API_VERSION", None)
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    def test_init_with_default_api_version(self, mock_azure_model, mock_get_config):
        """Test initialization with default API version when not configured."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")

        assert llm.api_version == "2024-05-01-preview"
        call_kwargs = mock_azure_model.call_args[1]
        assert call_kwargs["api_version"] == "2024-05-01-preview"

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
        "https://test.openai.azure.com/",
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    def test_init_strips_endpoint_trailing_slash(
        self, mock_azure_model, mock_get_config
    ):
        """Test that endpoint trailing slash is removed."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")

        assert llm.endpoint == "https://test.openai.azure.com"
        call_kwargs = mock_azure_model.call_args[1]
        assert call_kwargs["endpoint"] == "https://test.openai.azure.com"

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
        "https://test.services.ai.azure.com",
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    def test_init_adds_models_suffix_for_ai_foundry(
        self, mock_azure_model, mock_get_config
    ):
        """Test that /models suffix is added for Azure AI Foundry endpoints."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")

        assert llm.endpoint == "https://test.services.ai.azure.com/models"
        call_kwargs = mock_azure_model.call_args[1]
        assert call_kwargs["endpoint"] == "https://test.services.ai.azure.com/models"

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
        "https://test.services.ai.azure.com/models",
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    def test_init_does_not_duplicate_models_suffix(
        self, mock_azure_model, mock_get_config
    ):
        """Test that /models suffix is not duplicated if already present."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")

        assert llm.endpoint == "https://test.services.ai.azure.com/models"
        call_kwargs = mock_azure_model.call_args[1]
        assert call_kwargs["endpoint"] == "https://test.services.ai.azure.com/models"

    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "http://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    def test_init_invalid_endpoint_raises_error(self, mock_get_config):
        """Test that non-HTTPS endpoint raises ValueError."""
        mock_get_config.return_value = {"model": "gpt-4"}

        with pytest.raises(ValueError) as exc_info:
            AzureLLM(name="TestAzure")

        assert "must start with 'https://'" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    async def test_generate_response_success_with_system_prompt(
        self, mock_azure_model, mock_get_config
    ):
        """Test successful response generation with system prompt."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        # Create mock response with metadata
        # Azure treats response_metadata as a dict-like object that also supports
        # attribute access
        mock_response = MagicMock()
        mock_response.text = "This is an Azure response"
        mock_response.id = "chatcmpl-12345"
        mock_response.response_metadata = DictWithAttr(
            {
                "model": "gpt-4",
                "token_usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                },
                "finish_reason": "stop",
            }
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", system_prompt="You are a helpful assistant.")
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Hello, Azure!"}
            ]
        )

        assert response == "This is an Azure response"

        # Verify metadata was extracted
        metadata = llm.get_last_response_metadata()
        assert metadata["response_id"] == "chatcmpl-12345"
        assert metadata["model"] == "gpt-4"
        assert metadata["provider"] == "azure"
        assert "timestamp" in metadata
        assert "response_time_seconds" in metadata
        assert metadata["usage"]["input_tokens"] == 10
        assert metadata["usage"]["output_tokens"] == 20
        assert metadata["usage"]["total_tokens"] == 30
        assert metadata["finish_reason"] == "stop"
        assert "raw_metadata" in metadata

    @pytest.mark.asyncio
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    async def test_generate_response_without_system_prompt(
        self, mock_azure_model, mock_get_config
    ):
        """Test response generation without system prompt."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        mock_response = MagicMock()
        mock_response.text = "Response without system prompt"
        mock_response.id = "chatcmpl-67890"
        mock_response.response_metadata = DictWithAttr({"model": "gpt-4"})

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")  # No system prompt
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Test message"}
            ]
        )

        assert response == "Response without system prompt"

        # Verify ainvoke was called with only HumanMessage (no SystemMessage)
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].content == "Test message"

    @pytest.mark.asyncio
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    async def test_generate_response_without_usage_metadata(
        self, mock_azure_model, mock_get_config
    ):
        """Test response when usage metadata is not available."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        # Response without usage in metadata
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-abc"
        mock_response.response_metadata = DictWithAttr({"model": "gpt-4"})
        # No token_usage in metadata

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    async def test_generate_response_without_response_metadata(
        self, mock_azure_model, mock_get_config
    ):
        """Test response when response_metadata attribute is missing."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        # Response without response_metadata attribute
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-xyz"
        del mock_response.response_metadata  # Remove attribute

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gpt-4"
        assert metadata["usage"] == {}
        assert metadata["finish_reason"] is None

    @pytest.mark.asyncio
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    async def test_generate_response_api_error(self, mock_azure_model, mock_get_config):
        """Test error handling when API call fails."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        # Simulate API error
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API rate limit exceeded"))
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Test message"}
            ]
        )

        # Should return error message instead of raising
        assert "Error generating response" in response
        assert "API rate limit exceeded" in response

        # Verify error metadata was stored
        metadata = llm.get_last_response_metadata()
        assert metadata["response_id"] is None
        assert metadata["model"] == "gpt-4"
        assert metadata["provider"] == "azure"
        assert "timestamp" in metadata
        assert "error" in metadata
        assert "API rate limit exceeded" in metadata["error"]
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    async def test_generate_response_404_error_with_helpful_message(
        self, mock_azure_model, mock_get_config
    ):
        """Test that 404 errors provide helpful error messages."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        # Simulate 404 error with proper exception class
        class AzureError(Exception):
            def __init__(self, message, status_code=None):
                super().__init__(message)
                self.status_code = status_code
                self.response = MagicMock()
                if status_code:
                    self.response.url = "https://test.openai.azure.com/models/gpt-4"

        error = AzureError("404 Resource not found", status_code=404)
        mock_llm.ainvoke = AsyncMock(side_effect=error)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Test message"}
            ]
        )

        # Should contain helpful error message
        assert "Error generating response" in response
        assert "404" in response or "Resource not found" in response
        assert "Model name" in response or "deployment name" in response

    @pytest.mark.asyncio
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    async def test_generate_response_tracks_timing(
        self, mock_azure_model, mock_get_config
    ):
        """Test that response timing is tracked correctly."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        mock_response = MagicMock()
        mock_response.text = "Timed response"
        mock_response.id = "chatcmpl-time"
        mock_response.response_metadata = DictWithAttr({"model": "gpt-4"})

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")
        await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        assert "response_time_seconds" in metadata
        assert isinstance(metadata["response_time_seconds"], (int, float))
        assert metadata["response_time_seconds"] >= 0

    def test_get_last_response_metadata_returns_copy(self):
        """Test that get_last_response_metadata returns a copy."""
        with patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key"):
            with patch(
                "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
                "https://test.openai.azure.com",
            ):
                with patch(
                    "llm_clients.azure_llm.Config.get_azure_config"
                ) as mock_get_config:
                    mock_get_config.return_value = {"model": "gpt-4"}
                    with patch(
                        "llm_clients.azure_llm.AzureAIChatCompletionsModel"
                    ) as mock_azure_model:
                        mock_llm = MagicMock()
                        mock_llm.model_name = "gpt-4"
                        mock_azure_model.return_value = mock_llm

                        llm = AzureLLM(name="TestAzure")
                        llm.last_response_metadata = {"test": "value"}

                        metadata1 = llm.get_last_response_metadata()
                        metadata2 = llm.get_last_response_metadata()

                        # Should be equal but not the same object
                        assert metadata1 == metadata2
                        assert metadata1 is not metadata2

                        # Modifying returned copy shouldn't affect internal state
                        metadata1["modified"] = True
                        assert "modified" not in llm.last_response_metadata

    def test_set_system_prompt(self):
        """Test set_system_prompt method."""
        with patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key"):
            with patch(
                "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
                "https://test.openai.azure.com",
            ):
                with patch(
                    "llm_clients.azure_llm.Config.get_azure_config"
                ) as mock_get_config:
                    mock_get_config.return_value = {"model": "gpt-4"}
                    with patch(
                        "llm_clients.azure_llm.AzureAIChatCompletionsModel"
                    ) as mock_azure_model:
                        mock_llm = MagicMock()
                        mock_llm.model_name = "gpt-4"
                        mock_azure_model.return_value = mock_llm

                        llm = AzureLLM(name="TestAzure", system_prompt="Initial prompt")
                        assert llm.system_prompt == "Initial prompt"

                        llm.set_system_prompt("Updated prompt")
                        assert llm.system_prompt == "Updated prompt"

    @pytest.mark.asyncio
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    async def test_generate_structured_response_success(
        self, mock_azure_model, mock_get_config
    ):
        """Test successful structured response generation."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        # Create a test Pydantic model
        class TestResponse(BaseModel):
            answer: str = Field(description="The answer")
            reasoning: str = Field(description="The reasoning")

        # Mock structured LLM
        mock_structured_llm = MagicMock()
        test_response = TestResponse(answer="Yes", reasoning="Because it's correct")
        mock_structured_llm.ainvoke = AsyncMock(return_value=test_response)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", system_prompt="Test prompt")
        response = await llm.generate_structured_response(
            "What is the answer?", TestResponse
        )

        assert isinstance(response, TestResponse)
        assert response.answer == "Yes"
        assert response.reasoning == "Because it's correct"

        # Verify metadata was stored
        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gpt-4"
        assert metadata["provider"] == "azure"
        assert metadata["structured_output"] is True
        assert "timestamp" in metadata
        assert "response_time_seconds" in metadata

    @pytest.mark.asyncio
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    async def test_generate_structured_response_error(
        self, mock_azure_model, mock_get_config
    ):
        """Test error handling in structured response generation."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        class TestResponse(BaseModel):
            answer: str

        # Mock structured LLM to raise error
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(
            side_effect=Exception("Structured output failed")
        )
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")

        with pytest.raises(RuntimeError) as exc_info:
            await llm.generate_structured_response("Test", TestResponse)

        assert "Error generating structured response" in str(exc_info.value)
        assert "Structured output failed" in str(exc_info.value)

        # Verify error metadata was stored
        metadata = llm.get_last_response_metadata()
        assert "error" in metadata
        assert "Structured output failed" in metadata["error"]

    @pytest.mark.asyncio
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    async def test_generate_response_with_conversation_history(
        self, mock_azure_model, mock_get_config
    ):
        """Test generate_response with conversation_history parameter."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        mock_response = MagicMock()
        mock_response.text = "Response with history"
        mock_response.id = "chatcmpl-history"
        mock_response.response_metadata = DictWithAttr(
            {
                "model": "gpt-4",
                "token_usage": {
                    "input_tokens": 50,
                    "output_tokens": 20,
                },
            }
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", system_prompt="Test")

        # Provide conversation history
        history = [
            {
                "turn": 1,
                "speaker": "persona",
                "input": "Start",
                "response": "Hello",
                "early_termination": False,
                "logging": {},
            },
            {
                "turn": 2,
                "speaker": "agent",
                "input": "Hello",
                "response": "Hi there",
                "early_termination": False,
                "logging": {},
            },
        ]

        response = await llm.generate_response(conversation_history=history)

        assert response == "Response with history"

        # Verify ainvoke was called with correct messages
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]

        # Should have: SystemMessage + 2 history messages
        assert len(messages) == 3

    @pytest.mark.asyncio
    @patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key")
    @patch(
        "llm_clients.azure_llm.Config.AZURE_ENDPOINT", "https://test.openai.azure.com"
    )
    @patch("llm_clients.azure_llm.Config.get_azure_config")
    @patch("llm_clients.azure_llm.AzureAIChatCompletionsModel")
    async def test_timestamp_format(self, mock_azure_model, mock_get_config):
        """Test that timestamp is in ISO format."""
        mock_get_config.return_value = {"model": "gpt-4"}
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.id = "chatcmpl-ts"
        mock_response.response_metadata = DictWithAttr({"model": "gpt-4"})

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure")
        await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        timestamp = metadata["timestamp"]

        # Verify it's a valid ISO format timestamp
        try:
            datetime.fromisoformat(timestamp)
            timestamp_valid = True
        except ValueError:
            timestamp_valid = False

        assert timestamp_valid
