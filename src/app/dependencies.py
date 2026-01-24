from functools import lru_cache

from fastapi import Depends

from config import Settings, get_settings
from app.services.hitl import PendingActionService
from app.services.orchestrator import OrchestratorService


def provide_settings() -> Settings:
    return get_settings()


@lru_cache
def _get_pending_action_service_cached(env: str) -> PendingActionService:
    settings = get_settings()
    return PendingActionService(settings=settings)


def provide_hitl_service(
    settings: Settings = Depends(provide_settings),
) -> PendingActionService:
    return _get_pending_action_service_cached(settings.environment)


@lru_cache
def _get_orchestrator_service_cached(env: str) -> OrchestratorService:
    settings = get_settings()
    hitl = _get_pending_action_service_cached(env)
    return OrchestratorService(settings=settings, hitl_service=hitl)


def provide_orchestrator(
    settings: Settings = Depends(provide_settings),
    hitl_service: PendingActionService = Depends(provide_hitl_service),
) -> OrchestratorService:
    return _get_orchestrator_service_cached(settings.environment)
