#!/usr/bin/env python3
"""
Ollama LLM implementation for llama-index that supports Llama 4.
"""

import os
import logging
import ollama
from typing import Any, List, Optional, Sequence, Dict
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.base.llms.base import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.callbacks import CallbackManager

DEFAULT_OLLAMA_MODEL = "llama4"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_MAX_TOKENS = 1024

logger = logging.getLogger(__name__)

class OllamaLLM(CustomLLM):
    """Ollama LLM integration, with support for Llama 4."""

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama LLM.
        
        Args:
            model_name: Name of the model to use in Ollama
            ollama_url: URL of the Ollama API
            temperature: Temperature for generation
            context_window: Maximum context window
            max_tokens: Maximum number of tokens to generate
            system_prompt: System prompt to use
            callback_manager: Callback manager
        """
        # Set the URL for the Ollama API
        os.environ["OLLAMA_HOST"] = ollama_url
        
        # Store parameters
        self.model_name = model_name
        self.temperature = temperature
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or ""
        self.kwargs = kwargs

        super().__init__(callback_manager=callback_manager)
        
        # Check if the model is available in Ollama
        self._check_model()

    def _check_model(self) -> None:
        """Check if the model is available in Ollama."""
        try:
            models = ollama.list()
            model_names = [model['name'].split(':')[0] for model in models['models']]
            
            if self.model_name not in model_names:
                logger.warning(
                    f"Model {self.model_name} not found in Ollama. "
                    f"Available models: {', '.join(model_names)}. "
                    f"Pulling the model..."
                )
                # Pull the model automatically
                ollama.pull(self.model_name)
        except Exception as e:
            logger.error(f"Failed to check Ollama models: {e}")
            logger.warning(
                "Make sure Ollama is installed and running. "
                "Install: curl -fsSL https://ollama.com/install.sh | sh"
            )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model_name,
            is_chat_model=True,
        )

    def _format_message(self, prompt: str) -> Dict[str, str]:
        """Format the message for Ollama."""
        return {
            "role": "user",
            "content": prompt,
        }

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete the prompt."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append(self._format_message(prompt))

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                **self.kwargs,
            )
            logger.info(f"Ollama response for model {self.model_name} received")
            
            return CompletionResponse(
                text=response["message"]["content"],
                raw=response,
            )
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return CompletionResponse(
                text=f"Error: {str(e)}",
                raw={"error": str(e)},
            )

    @llm_completion_callback()
    def chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Chat with the Ollama model."""
        ollama_messages = []
        
        # Extract the system message if present
        system_message = self.system_prompt
        non_system_messages = []
        
        for message in messages:
            if message.role == "system":
                # If there are multiple system messages, concatenate them
                if system_message:
                    system_message += "\n" + message.content
                else:
                    system_message = message.content
            else:
                non_system_messages.append(message)
        
        # Add the system message
        if system_message:
            ollama_messages.append({"role": "system", "content": system_message})
        
        # Add the remaining messages
        for message in non_system_messages:
            ollama_messages.append({
                "role": message.role,
                "content": message.content,
            })

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=ollama_messages,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                **self.kwargs,
            )
            logger.info(f"Ollama chat response for model {self.model_name} received")
            
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=response["message"]["content"],
                ),
                raw=response,
            )
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=f"Error: {str(e)}",
                ),
                raw={"error": str(e)},
            )

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Stream the completion."""
        raise NotImplementedError("Stream completion not supported for Ollama LLM.")

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Stream the chat response."""
        raise NotImplementedError("Stream chat not supported for Ollama LLM.")