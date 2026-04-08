from typing import Dict, Tuple
from models import Issue

TASKS = {
    "easy": {
        "description": "Triage the single issue in the inbox by reading its description and adding the appropriate label. It is clearly a bug. Then 'submit'.",
        "issues": [
            Issue(
                id="ISSUE-101",
                title="App crashes on startup",
                description="When I click the application icon on my Android device, it shows a white screen and then immediately returns to the home screen. I am on Android 13.",
            )
        ],
        "max_steps": 5
    },
    "medium": {
        "description": "Identify and close the duplicate issue.",
        "issues": [
            Issue(id="ISSUE-201", title="Typo", description="welcome -> wellcome"),
            Issue(id="ISSUE-202", title="Payment failure", description="Stripe 400 EU"),
            Issue(id="ISSUE-203", title="EU payment issue", description="Stripe 400 EU"),
        ],
        "max_steps": 10
    },
    "hard": {
        "description": "Full triage of 5 issues.",
        "issues": [
            Issue(id="ISSUE-301", title="DB timeout", description="504 timeout"),
            Issue(id="ISSUE-302", title="Dark mode", description="Add dark mode"),
            Issue(id="ISSUE-303", title="Night theme", description="Duplicate dark mode"),
            Issue(id="ISSUE-304", title="NPE error", description="Null pointer"),
            Issue(id="ISSUE-305", title="Avatar limit", description="Increase size"),
        ],
        "max_steps": 25
    }
}

# ---------------- SAFE CLAMP ----------------
def _safe_score(val: float) -> float:
    # Strictly between (0,1)
    if val >= 1.0:
        return 0.99
    if val <= 0.0:
        return 0.01
    return float(val)

# ---------------- EASY ----------------
def grade_easy(issues: Dict[str, Issue]) -> float:
    issue = issues.get("ISSUE-101")
    if not issue:
        return 0.01
    return 0.99 if "bug" in issue.labels else 0.01

# ---------------- MEDIUM ----------------
def grade_medium(issues: Dict[str, Issue]) -> float:
    issue = issues.get("ISSUE-203")
    if not issue:
        return 0.01

    if issue.status == "duplicate":
        if issue.duplicate_of == "ISSUE-202":
            return 0.99
        return 0.5  # valid partial

    return 0.01

# ---------------- HARD ----------------
def grade_hard(issues: Dict[str, Issue]) -> float:
    score = 0.0
    max_score = 5.0

    def add(val):
        nonlocal score
        score += val

    # ISSUE-301
    i301 = issues.get("ISSUE-301")
    if i301:
        if "bug" in i301.labels: add(0.33)
        if i301.assignee == "backend": add(0.33)
        if i301.priority == "high": add(0.34)

    # ISSUE-302
    i302 = issues.get("ISSUE-302")
    if i302:
        if "feature" in i302.labels: add(0.33)
        if i302.assignee == "frontend": add(0.33)
        if i302.priority == "low": add(0.34)

    # ISSUE-303 duplicate
    i303 = issues.get("ISSUE-303")
    if i303:
        if i303.status == "duplicate":
            if i303.duplicate_of == "ISSUE-302":
                add(0.99)
            else:
                add(0.5)

    # ISSUE-304
    i304 = issues.get("ISSUE-304")
    if i304:
        if "bug" in i304.labels: add(0.33)
        if i304.assignee == "backend": add(0.33)
        if i304.priority == "high": add(0.34)

    # ISSUE-305
    i305 = issues.get("ISSUE-305")
    if i305:
        if "feature" in i305.labels: add(0.33)
        if i305.assignee == "frontend": add(0.33)
        if i305.priority == "low": add(0.34)

    # Normalize safely
    final_score = score / max_score

    return _safe_score(final_score)

# ---------------- GRADERS ----------------
GRADERS = {
    "easy": lambda issues: _safe_score(grade_easy(issues)),
    "medium": lambda issues: _safe_score(grade_medium(issues)),
    "hard": lambda issues: _safe_score(grade_hard(issues)),
}

# ---------------- TASK SETUP ----------------
def get_task_setup(task_name: str) -> Tuple[str, Dict[str, Issue], int]:
    task = TASKS[task_name]
    copied_issues = {i.id: Issue(**i.model_dump()) for i in task["issues"]}
    return task["description"], copied_issues, task["max_steps"]