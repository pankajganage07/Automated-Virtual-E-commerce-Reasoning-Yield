from config import Settings


class MemoryService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def fetch_similar_incidents(self, query: str) -> list[dict]:
        return []

    async def save_incident(self, summary: str) -> None:
        return None
