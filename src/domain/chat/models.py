from enum import Enum
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

class AgentState(str, Enum):
    THINKING = "thinking"     # Analyzing query
    RETRIEVING = "retrieving" # Fetching docs
    ANALYZING = "analyzing"   # Checking retrieval methods
    GENERATING = "generating" # Writing answer
    REFUSING = "refusing"     # Declining
    CLARIFYING = "clarifying" # Asking user
    DONE = "done"

# --- Actions ---
class RetrieveAction(BaseModel):
    action_type: Literal["retrieve"] = "retrieve"
    query: str
    rationale: str

class SummarizeAction(BaseModel):
    action_type: Literal["summarize"] = "summarize"
    doc_ids: Optional[List[str]] = None
    rationale: str

class ClarifyAction(BaseModel):
    action_type: Literal["clarify"] = "clarify"
    question: str
    rationale: str

class RefuseAction(BaseModel):
    action_type: Literal["refuse"] = "refuse"
    reason: str
    rationale: str

class AnswerAction(BaseModel):
    action_type: Literal["answer"] = "answer"
    answer: str
    confidence_score: float
    citations: List[str]
    rationale: str

# Union for polymorphic parsing
AgentAction = Union[RetrieveAction, SummarizeAction, ClarifyAction, RefuseAction, AnswerAction]

# --- History ---
class AgentStep(BaseModel):
    step_id: UUID = Field(default_factory=uuid4)
    state: AgentState
    thought: str
    action: Optional[AgentAction] = None
    observation: Optional[str] = None # Output of tool
    timestamp: float = 0.0
