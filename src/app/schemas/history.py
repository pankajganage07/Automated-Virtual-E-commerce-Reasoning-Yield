"""
Schemas for history/incidents endpoints.
"""

from pydantic import BaseModel, Field


class IncidentItem(BaseModel):
    """A single incident record."""

    id: int | None = Field(None, description="Incident ID")
    incident_summary: str = Field(..., description="Summary of the incident")
    root_cause: str | None = Field(None, description="Identified root cause")
    action_taken: str | None = Field(None, description="Actions that were taken")
    outcome: str | None = Field(None, description="Result of the actions")
    score: float | None = Field(None, description="Similarity score (for search results)")
    created_at: str | None = Field(None, description="When the incident was recorded")


class IncidentListResponse(BaseModel):
    """Response for listing incidents with pagination."""

    incidents: list[IncidentItem] = Field(..., description="List of incidents")
    total: int = Field(..., description="Total number of incidents")
    limit: int = Field(..., description="Number of items per page")
    offset: int = Field(..., description="Current offset")


class IncidentSearchResponse(BaseModel):
    """Response for semantic search of incidents."""

    query: str = Field(..., description="The search query used")
    results: list[IncidentItem] = Field(..., description="Matching incidents sorted by similarity")
    total_found: int = Field(..., description="Number of matches found")
