---
title: OpenEnv Issue Triage
emoji: 🐛
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# OpenEnv: IT Issue Tracker Triage Automation

This is a simulated real-world workflow environment for OpenEnv, evaluating an agent's ability to act as a Level 1 Support / Triage Engineer.

## Environment Description & Motivation
In software engineering, technical support systems are frequently flooded with unstructured bug reports and feature requests. Humans spend an enormous amount of time classifying these tickets, assigning them to the right teams, and identifying duplicates. This environment tasks an AI agent with performing this entire workflow directly on a mock issue tracker by reading reports and issuing structured actions. It represents a significantly more real-world applicable task than standard toy games or gridworlds.

## State Space
The underlying environment state holds a dictionary of `Issue` entries, each detailing an `id`, `title`, `description`, `status` ("open", "closed", "duplicate"), `labels`, `priority`, and an `assignee`.
It tracks the maximum steps per episode and the agent's current iteration count. 

## Action Space
The agent can perform the following Pydantic-typed actions natively (no string parsing required on the environment side other than JSON loading overhead in REST):
- **ViewIssueAction**: `{"action_type": "view_issue", "issue_id": "<id>"}`
- **AddLabelAction**: `{"action_type": "add_label", "issue_id": "<id>", "label": "bug/feature/duplicate/etc"}`
- **SetPriorityAction**: `{"action_type": "set_priority", "issue_id": "<id>", "priority": "low|medium|high|critical"}`
- **AssignIssueAction**: `{"action_type": "assign_issue", "issue_id": "<id>", "assignee": "frontend|backend|etc"}`
- **CloseIssueAction**: `{"action_type": "close_issue", "issue_id": "<id>", "reason": "duplicate|completed", "duplicate_of": "<id>"}`
- **SubmitAction**: `{"action_type": "submit"}` -> Concludes the episode.

## Observation Space
The agent receives:
- **inbox**: A list of truncated issue summaries (hiding the full description until `view_issue` is utilized).
- **last_action_result**: The string result or error feedback from the prior attempted action.
- **step_count**: Current iteration.
- **max_steps**: The step limit before automatic termination.
- **task_description**: Detailed natural language instructions for what the agent must achieve.

## Task Difficulty Breakdown
1. **Easy:** A single issue is provided. The agent must add the label "bug" based on reading a clear application crash description.
2. **Medium:** Three issues are provided where two express the exact same payment error. The agent must deduplicate the latter issue against the former one using the `close_issue` action.
3. **Hard:** A full triage batch. Five distinct issues. The agent must label "bug" and assign "backend", label "feature" and assign "frontend", scale priority appropriately, and find duplicates among them. Requires ~10-15 correct actions to achieve high rewards.

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run baseline inference script:
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4-turbo"
export HF_TOKEN="your_openai_api_key_or_hf_token"

python inference.py
```

3. Docker deployment (used for HF Space):
```bash
docker build -t openenv-tracker .
docker run -p 7860:7860 openenv-tracker
```
The FastAPI instance will reply to `POST /reset` and `POST /step` fulfilling standard evaluation pings.

## Baseline scores
On a typical frontier-class model (e.g., GPT-4o or Claude 3.5 Sonnet equivalents):
- Easy: 1.0
- Medium: 1.0 
- Hard: ~0.8-1.0 (sometimes struggles to assign every component flawlessly if it rushes to submit).
