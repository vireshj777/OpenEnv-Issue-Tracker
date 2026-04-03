import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Ensure we can import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import IssueTriageEnv
from models import Observation, Action, Reward, State
from models import ViewIssueAction, AddLabelAction, SetPriorityAction, AssignIssueAction, CloseIssueAction, SubmitAction

app = FastAPI(title="OpenEnv Issue Tracker", description="Issue Triage environment for OpenEnv evaluation")

# Instantiate a single global environment for the space
env = IssueTriageEnv()

class ResetRequest(BaseModel):
    task: str = "easy"

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]

@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest = ResetRequest()):
    try:
        obs = env.reset(req.task)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResponse)
def step_env(action: Dict[str, Any]):
    # Since FastAPI will parse the generic dict, we manually parse into the Union based on action_type
    try:
        action_type = action.get("action_type")
        if action_type == "view_issue":
            act = ViewIssueAction(**action)
        elif action_type == "add_label":
            act = AddLabelAction(**action)
        elif action_type == "set_priority":
            act = SetPriorityAction(**action)
        elif action_type == "assign_issue":
            act = AssignIssueAction(**action)
        elif action_type == "close_issue":
            act = CloseIssueAction(**action)
        elif action_type == "submit":
            act = SubmitAction(**action)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")
            
        obs, reward, done, info = env.step(act)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=State)
def get_state():
    try:
        return env.get_state()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "API is running"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
