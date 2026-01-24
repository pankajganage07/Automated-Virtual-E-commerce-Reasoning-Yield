from __future__ import annotations

import hashlib
import random
from typing import Sequence

from openai import AsyncOpenAI, OpenAIError

from config import Settings


class EmbeddingProvider:
    """
    Wraps OpenAI embeddings with a deterministic fallback when API keys are absent.
    """

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.openai_api_key
        self._model = settings.embedding_model
        self._client = AsyncOpenAI(api_key=self._api_key) if self._api_key else None
        self._dim = 1536  # matches text-embedding-3-small

    async def embed(self, text: str) -> list[float]:
        if self._client:
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=text,
                )
                return response.data[0].embedding
            except OpenAIError:
                pass  # fall back

        return self._fallback_embed(text)

    def _fallback_embed(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        return [rng.uniform(-1, 1) for _ in range(self._dim)]
