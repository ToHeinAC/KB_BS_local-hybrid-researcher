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

    def generate_structured_with_language(
        self,
        prompt: str,
        response_model: type[T],
        target_language: str,
        max_retries: int = 2,
    ) -> T:
        """Generate structured output with language enforcement.

        If output contains significant other-language content, retry with
        stronger language instruction.

        Args:
            prompt: The prompt to send
            response_model: Pydantic model for response
            target_language: Target language code ('de' or 'en')
            max_retries: Number of retries for language validation

        Returns:
            Parsed response as Pydantic model instance
        """
        result = self.generate_structured(prompt, response_model)

        # Validate language
        if not self._validate_language(result, target_language):
            logger.warning(
                f"Language validation failed for target '{target_language}', retrying"
            )
            # Retry with explicit language enforcement
            lang_name = "German" if target_language == "de" else "English"
            enforced_prompt = f"""CRITICAL LANGUAGE REQUIREMENT: You MUST respond ONLY in {lang_name}.
Do NOT use any other language. All text in your response must be in {lang_name}.
WICHTIG/IMPORTANT: Antworte NUR auf {lang_name}.

{prompt}"""
            result = self.generate_structured(enforced_prompt, response_model)

        return result

    def _validate_language(self, result: BaseModel, target: str) -> bool:
        """Check if result is primarily in target language.

        Uses a simple heuristic based on common language markers.

        Args:
            result: Pydantic model result to validate
            target: Target language code ('de' or 'en')

        Returns:
            True if result appears to be in target language
        """
        # Get text representation
        text = str(result.model_dump())
        text_lower = text.lower()

        # Common German markers (weighted by specificity)
        german_markers = [
            " der ", " die ", " das ", " und ", " ist ", " für ", " mit ",
            " ein ", " eine ", " einer ", " einem ", " den ", " dem ",
            " auf ", " bei ", " nach ", " über ", " unter ", " durch ",
            " wird ", " werden ", " wurde ", " wurden ", " sind ", " waren ",
            " kann ", " können ", " muss ", " müssen ", " soll ", " sollte ",
            " nicht ", " auch ", " oder ", " aber ", " wenn ", " dann ",
        ]
        # Common English markers
        english_markers = [
            " the ", " and ", " is ", " for ", " with ", " of ", " to ",
            " a ", " an ", " in ", " on ", " at ", " by ", " from ",
            " that ", " this ", " are ", " was ", " were ", " have ",
            " has ", " had ", " will ", " would ", " can ", " could ",
            " should ", " may ", " might ", " must ", " shall ",
        ]

        # Count occurrences (not just presence)
        german_count = sum(text_lower.count(m) for m in german_markers)
        english_count = sum(text_lower.count(m) for m in english_markers)

        # If very little text, assume valid
        if german_count + english_count < 3:
            return True

        if target == "de":
            return german_count >= english_count
        else:
            return english_count >= german_count
