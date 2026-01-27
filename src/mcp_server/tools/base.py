from __future__ import annotations

import abc
from typing import Any, Dict

from pydantic import BaseModel, ValidationError

from mcp_server.db import get_session


class BaseTool(abc.ABC):
    name: str

    async def __call__(self, arguments: dict[str, Any]) -> Any:
        schema = self.request_model()
        try:
            payload = schema(**arguments)
        except ValidationError as exc:
            raise ValueError(f"Invalid arguments for {self.name}: {exc}") from exc

        async with get_session() as session:
            return await self.run(session=session, payload=payload)

    @abc.abstractmethod
    def request_model(self) -> type[BaseModel]: ...

    @abc.abstractmethod
    async def run(self, session, payload) -> Any: ...
