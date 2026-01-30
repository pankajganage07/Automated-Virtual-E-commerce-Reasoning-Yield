"""
API client for Avery OpsBrain backend.

Wraps all backend endpoints with proper error handling and typing.
"""

from __future__ import annotations

import httpx
from dataclasses import dataclass, field
from typing import Any

from .config import FrontendConfig, get_config


@dataclass
class PendingAction:
    """A pending action awaiting approval."""

    id: int
    agent: str
    action_type: str
    payload: dict[str, Any]
    reasoning: str
    status: str

    @classmethod
    def from_dict(cls, data: dict) -> "PendingAction":
        return cls(
            id=data["id"],
            agent=data.get("agent", data.get("agent_name", "unknown")),
            action_type=data["action_type"],
            payload=data.get("payload", {}),
            reasoning=data.get("reasoning", ""),
            status=data.get("status", "pending"),
        )


@dataclass
class QueryResponse:
    """Response from a query to the backend."""

    answer: str
    diagnostics: list[str]
    pending_actions: list[PendingAction]
    thread_id: str | None
    hitl_waiting: bool
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "QueryResponse":
        return cls(
            answer=data.get("answer", ""),
            diagnostics=data.get("diagnostics", []),
            pending_actions=[PendingAction.from_dict(a) for a in data.get("pending_actions", [])],
            thread_id=data.get("thread_id"),
            hitl_waiting=data.get("hitl_waiting", False),
            raw=data,
        )


@dataclass
class ActionResult:
    """Result of an action approval/execution."""

    action_id: int
    status: str
    success: bool
    message: str
    result: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ActionResult":
        return cls(
            action_id=data.get("action_id", 0),
            status=data.get("status", "unknown"),
            success=data.get("success", False),
            message=data.get("message", ""),
            result=data.get("result"),
        )


@dataclass
class HistoryItem:
    """An incident from memory/history."""

    id: int | None
    incident_summary: str
    root_cause: str | None
    action_taken: str | None
    outcome: str | None
    score: float | None = None
    created_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "HistoryItem":
        return cls(
            id=data.get("id"),
            incident_summary=data.get("incident_summary", ""),
            root_cause=data.get("root_cause"),
            action_taken=data.get("action_taken"),
            outcome=data.get("outcome"),
            score=data.get("score"),
            created_at=data.get("created_at"),
        )


class APIError(Exception):
    """Error from the API."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")


class APIClient:
    """Client for interacting with the Avery OpsBrain backend."""

    def __init__(self, config: FrontendConfig | None = None):
        self.config = config or get_config()
        self._client = httpx.Client(
            base_url=self.config.api_base_url,
            timeout=self.config.api_timeout,
        )

    def _handle_response(self, response: httpx.Response) -> dict:
        """Handle response and raise APIError on failure."""
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise APIError(response.status_code, detail)
        return response.json()

    # =========================================================================
    # Query Endpoints
    # =========================================================================

    def query(
        self,
        question: str,
        thread_id: str | None = None,
        history: list[dict] | None = None,
    ) -> QueryResponse:
        """
        Send a query to the backend.

        Args:
            question: The user's question
            thread_id: Optional thread ID for conversation continuity
            history: Optional conversation history

        Returns:
            QueryResponse with answer and any pending actions
        """
        payload: dict[str, Any] = {"question": question}

        metadata: dict[str, Any] = {}
        if thread_id:
            metadata["thread_id"] = thread_id
        if history:
            metadata["history"] = history
        if metadata:
            payload["metadata"] = metadata

        response = self._client.post("/query/", json=payload)
        data = self._handle_response(response)
        return QueryResponse.from_dict(data)

    def resume_query(
        self,
        thread_id: str,
        approved_action_ids: list[int],
        rejected_action_ids: list[int] | None = None,
    ) -> QueryResponse:
        """
        Resume a paused query after HITL approval.

        Args:
            thread_id: The thread ID from the original query
            approved_action_ids: IDs of approved actions
            rejected_action_ids: IDs of rejected actions

        Returns:
            QueryResponse with final answer
        """
        payload = {
            "thread_id": thread_id,
            "approved_action_ids": approved_action_ids,
            "rejected_action_ids": rejected_action_ids or [],
        }
        response = self._client.post("/query/resume", json=payload)
        data = self._handle_response(response)
        return QueryResponse.from_dict(data)

    # =========================================================================
    # Action Endpoints (HITL)
    # =========================================================================

    def get_pending_actions(self) -> list[PendingAction]:
        """Get all pending actions awaiting approval."""
        response = self._client.get("/actions/pending")
        data = self._handle_response(response)
        return [PendingAction.from_dict(a) for a in data.get("items", [])]

    def approve_action(
        self,
        action_id: int,
        execute_immediately: bool = True,
        comment: str | None = None,
    ) -> ActionResult:
        """
        Approve a pending action.

        Args:
            action_id: The action ID to approve
            execute_immediately: If True, execute the action immediately
            comment: Optional comment

        Returns:
            ActionResult with execution result if execute_immediately=True
        """
        payload = {
            "status": "approved",
            "execute_immediately": execute_immediately,
        }
        if comment:
            payload["comment"] = comment

        response = self._client.post(f"/actions/approve/{action_id}", json=payload)
        data = self._handle_response(response)
        return ActionResult.from_dict(data)

    def reject_action(
        self,
        action_id: int,
        comment: str | None = None,
    ) -> ActionResult:
        """
        Reject a pending action.

        Args:
            action_id: The action ID to reject
            comment: Optional rejection reason

        Returns:
            ActionResult confirming rejection
        """
        payload = {"status": "rejected"}
        if comment:
            payload["comment"] = comment

        response = self._client.post(f"/actions/approve/{action_id}", json=payload)
        data = self._handle_response(response)
        return ActionResult.from_dict(data)

    def execute_action(self, action_id: int) -> ActionResult:
        """
        Execute an already-approved action.

        Args:
            action_id: The action ID to execute

        Returns:
            ActionResult with execution result
        """
        response = self._client.post(f"/actions/execute/{action_id}")
        data = self._handle_response(response)
        return ActionResult.from_dict(data)

    # =========================================================================
    # History Endpoints
    # =========================================================================

    def list_history(
        self,
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[list[HistoryItem], int]:
        """
        List past incidents from memory.

        Args:
            limit: Max items to return (1-50)
            offset: Pagination offset

        Returns:
            Tuple of (items, total_count)
        """
        response = self._client.get(
            "/history/",
            params={"limit": limit, "offset": offset},
        )
        data = self._handle_response(response)
        items = [HistoryItem.from_dict(i) for i in data.get("incidents", [])]
        return items, data.get("total", len(items))

    def search_history(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[HistoryItem]:
        """
        Semantic search for similar past incidents.

        Args:
            query: Search query (min 3 chars)
            top_k: Number of results (1-10)

        Returns:
            List of matching incidents with similarity scores
        """
        response = self._client.get(
            "/history/search",
            params={"query": query, "top_k": top_k},
        )
        data = self._handle_response(response)
        return [HistoryItem.from_dict(i) for i in data.get("results", [])]

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> bool:
        """Check if the backend is healthy."""
        try:
            response = self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
