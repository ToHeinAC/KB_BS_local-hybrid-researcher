"""Ollama client with structured output and retry logic."""

import json
import logging
from typing import TypeVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ContextOverflowError(Exception):
    """Raised when context exceeds safe limit."""

    pass


class StructuredOutputError(Exception):
    """Raised when structured output parsing fails."""

    pass


class OllamaClient:
    """Client for Ollama with structured output and retry logic."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        num_ctx: int | None = None,
    ):
        """Initialize Ollama client.

        Args:
            model: Model name (defaults to settings)
            base_url: Ollama base URL (defaults to settings)
            num_ctx: Context window size (defaults to settings)
        """
        self.model = model or settings.ollama_model
        self.fallback_model = settings.ollama_fallback_model
        self.base_url = base_url or settings.ollama_base_url
        self.num_ctx = num_ctx or settings.ollama_num_ctx
        self.safe_limit = settings.safe_context_limit

        self._llm: ChatOllama | None = None
        self._fallback_llm: ChatOllama | None = None

        logger.info(
            f"OllamaClient initialized: model={self.model}, "
            f"num_ctx={self.num_ctx}, safe_limit={self.safe_limit}"
        )

    @property
    def llm(self) -> ChatOllama:
        """Get or create primary LLM instance."""
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                num_ctx=self.num_ctx,
                temperature=0,
                timeout=60,  # 60 second timeout to prevent UI hangs
            )
        return self._llm

    @property
    def fallback_llm(self) -> ChatOllama:
        """Get or create fallback LLM instance."""
        if self._fallback_llm is None:
            self._fallback_llm = ChatOllama(
                model=self.fallback_model,
                base_url=self.base_url,
                num_ctx=self.num_ctx,
                temperature=0,
                timeout=60,  # 60 second timeout to prevent UI hangs
            )
        return self._fallback_llm

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation).

        Uses ~4 characters per token as a rough estimate.
        """
        return len(text) // 4

    def check_context_limit(self, prompt: str) -> None:
        """Check if prompt exceeds safe context limit.

        Args:
            prompt: The prompt to check

        Raises:
            ContextOverflowError: If prompt exceeds safe limit
        """
        estimated_tokens = self.estimate_tokens(prompt)
        if estimated_tokens > self.safe_limit:
            raise ContextOverflowError(
                f"Estimated tokens ({estimated_tokens}) exceeds "
                f"safe limit ({self.safe_limit})"
            )

    @retry(
        retry=retry_if_exception_type((json.JSONDecodeError, ValidationError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        use_fallback: bool = False,
    ) -> T:
        """Generate structured output using json_mode.

        Args:
            prompt: The prompt to send
            response_model: Pydantic model for response
            use_fallback: Whether to use fallback model

        Returns:
            Parsed response as Pydantic model instance

        Raises:
            ContextOverflowError: If prompt exceeds safe limit
            StructuredOutputError: If parsing fails after retries
        """
        self.check_context_limit(prompt)

        llm = self.fallback_llm if use_fallback else self.llm

        # Create structured output with json_mode (required for Ollama <30B)
        structured_llm = llm.with_structured_output(
            response_model,
            method="json_mode",
        )

        # Build prompt with JSON schema hint
        schema_hint = response_model.model_json_schema()
        full_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{json.dumps(schema_hint, indent=2)}

JSON response:"""

        try:
            result = structured_llm.invoke(full_prompt)
            if isinstance(result, response_model):
                return result
            # If result is a dict, validate it
            return response_model.model_validate(result)

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Structured output failed, retrying: {e}")
            raise

    def generate_structured_safe(
        self,
        prompt: str,
        response_model: type[T],
    ) -> T:
        """Generate structured output with fallback on failure.

        Tries primary model first, falls back to smaller model if needed.

        Args:
            prompt: The prompt to send
            response_model: Pydantic model for response

        Returns:
            Parsed response as Pydantic model instance
        """
        try:
            return self.generate_structured(prompt, response_model, use_fallback=False)
        except RetryError:
            logger.warning("Primary model failed, trying fallback")
            try:
                return self.generate_structured(prompt, response_model, use_fallback=True)
            except RetryError as e:
                raise StructuredOutputError(
                    f"Both primary and fallback models failed: {e}"
                ) from e

    def generate(self, prompt: str, use_fallback: bool = False) -> str:
        """Generate unstructured text response.

        Args:
            prompt: The prompt to send
            use_fallback: Whether to use fallback model

        Returns:
            Generated text

        Raises:
            ContextOverflowError: If prompt exceeds safe limit
        """
        self.check_context_limit(prompt)

        llm = self.fallback_llm if use_fallback else self.llm
        response = llm.invoke(prompt)
        return response.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def generate_with_retry(self, prompt: str) -> str:
        """Generate text with retry logic.

        Args:
            prompt: The prompt to send

        Returns:
            Generated text
        """
        return self.generate(prompt)

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            self.llm.invoke("test")
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
