from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class InvokeRequest(BaseModel):
    tool: str = Field(..., description="Tool name to execute, e.g., 'get_sales_summary'.")
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolMetadata(BaseModel):
    tool: str
    duration_ms: float


class SuccessResponse(BaseModel):
    success: Literal[True] = True
    result: dict[str, Any] | list[Any] | Any
    metadata: ToolMetadata


class ErrorDetail(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail
