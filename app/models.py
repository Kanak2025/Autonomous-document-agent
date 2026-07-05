from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class AgentRequest(BaseModel):
    request: str = Field(..., description="Natural language request from the user")

    @field_validator("request")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("request must not be empty")
        return v.strip()


class TaskItem(BaseModel):
    id: int
    name: str
    section: str
    status: str = "pending"  # pending -> done -> failed


class ExecutionPlan(BaseModel):
    document_type: str
    title: str
    assumptions: List[str] = []
    sections: List[str]
    tasks: List[TaskItem]


class ReflectionResult(BaseModel):
    passed: bool
    issues: List[str] = []
    revised_sections: List[str] = []


class AgentResponse(BaseModel):
    status: str
    message: str
    plan: ExecutionPlan
    reflection: ReflectionResult
    document_filename: str
    download_url: str
    section_word_counts: Dict[str, int] = {}
    engine_meta: Dict[str, Any] = {}
    rate_limit: Optional[Dict[str, Any]] = None