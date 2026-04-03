from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Union

class Issue(BaseModel):
    id: str
    title: str
    description: str
    status: Literal["open", "closed", "duplicate"] = "open"
    labels: List[str] = Field(default_factory=list)
    priority: Optional[str] = None
    assignee: Optional[str] = None
    duplicate_of: Optional[str] = None

class State(BaseModel):
    current_task_id: str
    issues: Dict[str, Issue]
    step_count: int
    max_steps: int
    is_done: bool
    last_action_result: str = ""

# --- Action Models ---
class ViewIssueAction(BaseModel):
    action_type: Literal["view_issue"] = "view_issue"
    issue_id: str

class AddLabelAction(BaseModel):
    action_type: Literal["add_label"] = "add_label"
    issue_id: str
    label: str

class SetPriorityAction(BaseModel):
    action_type: Literal["set_priority"] = "set_priority"
    issue_id: str
    priority: Literal["low", "medium", "high", "critical"]

class AssignIssueAction(BaseModel):
    action_type: Literal["assign_issue"] = "assign_issue"
    issue_id: str
    assignee: str

class CloseIssueAction(BaseModel):
    action_type: Literal["close_issue"] = "close_issue"
    issue_id: str
    reason: Literal["completed", "duplicate", "wont_fix"]
    duplicate_of: Optional[str] = None

class SubmitAction(BaseModel):
    action_type: Literal["submit"] = "submit"

Action = Union[
    ViewIssueAction,
    AddLabelAction,
    SetPriorityAction,
    AssignIssueAction,
    CloseIssueAction,
    SubmitAction
]

# --- Observation & Reward Models ---
class IssueSummary(BaseModel):
    id: str
    title: str
    status: str
    labels: List[str]
    priority: Optional[str]
    assignee: Optional[str]

class Observation(BaseModel):
    inbox: List[IssueSummary]
    last_action_result: str
    step_count: int
    max_steps: int
    task_description: str

class Reward(BaseModel):
    value: float
    reason: str
