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
        "description": "Identify and close the duplicate issue. You have 3 issues. Two of them describe the same problem. Close the one that was filed later (higher ID) as a duplicate of the earlier one (lower ID). Then 'submit'.",
        "issues": [
            Issue(
                id="ISSUE-201",
                title="Typo on the landing page",
                description="The word 'welcome' is misspelled as 'wellcome' on the main page.",
            ),
            Issue(
                id="ISSUE-202",
                title="Payment gateway failure",
                description="Users from EU cannot checkout using credit cards. The Stripe API returns a 400 bad request error.",
            ),
            Issue(
                id="ISSUE-203",
                title="Can't process payments in Europe",
                description="Credit card payments are failing for European customers. Console shows Stripe 400 error.",
            )
        ],
        "max_steps": 10
    },
    "hard": {
        "description": "Full triage of 5 issues. Read them carefully. 1) Label bugs as 'bug' and assign to 'backend', set priority 'high'. 2) Label feature requests as 'feature' and assign to 'frontend', set priority 'low'. 3) Identify and close any duplicates of existing issues. Then 'submit'.",
        "issues": [
            Issue(
                id="ISSUE-301",
                title="Database timeout on user login",
                description="The login API endpoint is returning a 504 Gateway Timeout after 30 seconds. Logs show the database is not responding to connection requests from the auth service.",
            ),
            Issue(
                id="ISSUE-302",
                title="Dark mode needed",
                description="It would be native looking and great if the app had a dark mode toggle in the settings menu.",
            ),
            Issue(
                id="ISSUE-303",
                title="Add a night theme",
                description="Please add a dark theme option to the settings. My eyes hurt at night.",
            ),
            Issue(
                id="ISSUE-304",
                title="NullPointerException in UserProfile fetching",
                description="Stack trace attached. The UserProfile service is throwing an NPE when a user without an avatar tries to load their profile page.",
            ),
            Issue(
                id="ISSUE-305",
                title="Increase avatar file size limit",
                description="Users should be able to upload avatars up to 5MB instead of 1MB.",
            )
        ],
        "max_steps": 25
    }
}

def grade_easy(issues: Dict[str, Issue]) -> float:
    issue = issues.get("ISSUE-101")
    if not issue:
        return 0.01
    if "bug" in issue.labels:
        return 0.99
    return 0.01

def grade_medium(issues: Dict[str, Issue]) -> float:
    issue = issues.get("ISSUE-203")
    if not issue:
        return 0.01
    if issue.status == "duplicate" and issue.duplicate_of == "ISSUE-202":
        return 0.99
    if issue.status == "duplicate":
        return 0.5  # Closed as duplicate but wrong target
    return 0.01

def grade_hard(issues: Dict[str, Issue]) -> float:
    score = 0.0
    max_score = 5.0
    
    # ISSUE-301: Bug, Backend, High
    i301 = issues.get("ISSUE-301")
    if i301:
        if "bug" in i301.labels: score += 0.33
        if i301.assignee == "backend": score += 0.33
        if i301.priority == "high": score += 0.34
        
    # ISSUE-302: Feature, Frontend, Low
    i302 = issues.get("ISSUE-302")
    if i302:
        if "feature" in i302.labels: score += 0.33
        if i302.assignee == "frontend": score += 0.33
        if i302.priority == "low": score += 0.34

    # ISSUE-303: Duplicate of 302
    i303 = issues.get("ISSUE-303")
    if i303:
        if i303.status == "duplicate" and i303.duplicate_of == "ISSUE-302": score += 1.0
        elif i303.status == "duplicate": score += 0.5

    # ISSUE-304: Bug, Backend, High
    i304 = issues.get("ISSUE-304")
    if i304:
        if "bug" in i304.labels: score += 0.33
        if i304.assignee == "backend": score += 0.33
        if i304.priority == "high": score += 0.34

    # ISSUE-305: Feature, Frontend, Low
    i305 = issues.get("ISSUE-305")
    if i305:
        if "feature" in i305.labels: score += 0.33
        if i305.assignee == "frontend": score += 0.33
        if i305.priority == "low": score += 0.34

    final_score = score / max_score
    if final_score >= 0.99:
        return 0.99
    if final_score <= 0.01:
        return 0.01
    return final_score

def _clamp(val: float) -> float:
    return max(0.01, min(0.99, float(val)))

GRADERS = {
    "easy": lambda issues: _clamp(grade_easy(issues)),
    "medium": lambda issues: _clamp(grade_medium(issues)),
    "hard": lambda issues: _clamp(grade_hard(issues))
}

def get_task_setup(task_name: str) -> Tuple[str, Dict[str, Issue], int]:
    task = TASKS[task_name]
    # Return deep copies to avoid state mutation across resets
    copied_issues = {i.id: Issue(**i.model_dump()) for i in task["issues"]}
    return task["description"], copied_issues, task["max_steps"]
