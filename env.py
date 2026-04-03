from typing import Tuple, Dict, Any
import copy
from models import Observation, Action, Reward, State, IssueSummary, Issue
from tasks import get_task_setup, GRADERS, TASKS

class IssueTriageEnv:
    def __init__(self):
        self.state: State = None
        
    def reset(self, task_name: str = "easy") -> Observation:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}")
            
        description, issues, max_steps = get_task_setup(task_name)
        
        self.state = State(
            current_task_id=task_name,
            issues=issues,
            step_count=0,
            max_steps=max_steps,
            is_done=False,
            last_action_result="Environment reset initialized."
        )
        return self._get_observation()

    def get_state(self) -> State:
        if not self.state:
            raise RuntimeError("Environment has not been initialized. Call reset() first.")
        return self.state

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if not self.state:
            raise RuntimeError("Environment has not been initialized. Call reset() first.")
            
        if self.state.is_done:
            return self._get_observation(), Reward(value=0.0, reason="Episode is already done."), True, {}

        self.state.step_count += 1
        reward_value = 0.0
        reason = ""
        
        try:
            # Process action
            if action.action_type == "view_issue":
                issue = self._get_issue(action.issue_id)
                self.state.last_action_result = f"Content of {action.issue_id}: TITLE: {issue.title} | DESCRIPTION: {issue.description}"
                reward_value, reason = 0.0, "Viewed issue safely."
                
            elif action.action_type == "add_label":
                issue = self._get_issue(action.issue_id)
                if action.label not in issue.labels:
                    issue.labels.append(action.label)
                    self.state.last_action_result = f"Added label '{action.label}' to {action.issue_id}."
                    reward_value, reason = 0.05, f"Added a label to {action.issue_id}."
                else:
                    self.state.last_action_result = f"Label '{action.label}' already exists on {action.issue_id}."
                    reward_value, reason = 0.0, "Label already existed."

            elif action.action_type == "set_priority":
                issue = self._get_issue(action.issue_id)
                issue.priority = action.priority
                self.state.last_action_result = f"Set priority '{action.priority}' on {action.issue_id}."
                reward_value, reason = 0.05, f"Set priority on {action.issue_id}."

            elif action.action_type == "assign_issue":
                issue = self._get_issue(action.issue_id)
                issue.assignee = action.assignee
                self.state.last_action_result = f"Assigned {action.issue_id} to '{action.assignee}'."
                reward_value, reason = 0.05, f"Assigned team to {action.issue_id}."

            elif action.action_type == "close_issue":
                issue = self._get_issue(action.issue_id)
                if issue.status != "open":
                    raise ValueError(f"Issue {action.issue_id} is already {issue.status}.")
                
                if action.reason == "duplicate":
                    if not action.duplicate_of:
                        raise ValueError("Must provide 'duplicate_of' ID when closing as duplicate.")
                    if action.duplicate_of not in self.state.issues:
                        raise ValueError(f"Target duplicate issue ID {action.duplicate_of} not found.")
                    if action.duplicate_of == action.issue_id:
                        raise ValueError("An issue cannot be a duplicate of itself.")
                    
                    issue.status = "duplicate"
                    issue.duplicate_of = action.duplicate_of
                    self.state.last_action_result = f"Closed {action.issue_id} as duplicate of {action.duplicate_of}."
                    reward_value, reason = 0.1, f"Closed {action.issue_id} as duplicate."
                else:
                    issue.status = "closed"
                    self.state.last_action_result = f"Closed {action.issue_id} with reason '{action.reason}'."
                    reward_value, reason = 0.1, f"Closed {action.issue_id}."

            elif action.action_type == "submit":
                self.state.last_action_result = "Task submitted by agent."
                self.state.is_done = True
                
                # Final evaluation
                grader = GRADERS[self.state.current_task_id]
                final_score = grader(self.state.issues)
                
                # We give the bulk of the reward at the end based on the grader
                reward_value = final_score
                reason = f"Final grade for task '{self.state.current_task_id}': {final_score}"

        except ValueError as e:
            self.state.last_action_result = f"Error: {str(e)}"
            reward_value, reason = -0.1, f"Invalid action: {str(e)}"

        if self.state.step_count >= self.state.max_steps and not self.state.is_done:
            self.state.is_done = True
            
            # evaluate even if max steps reached
            grader = GRADERS[self.state.current_task_id]
            final_score = grader(self.state.issues)
            reward_value += final_score
            reason += f" | Max steps reached. Final grade: {final_score}"
            self.state.last_action_result += " | Max steps reached. Auto-submitting."

        obs = self._get_observation()
        reward = Reward(value=reward_value, reason=reason)
        
        info = {
            "current_score": GRADERS[self.state.current_task_id](self.state.issues)
        }
        
        return obs, reward, self.state.is_done, info

    def _get_issue(self, issue_id: str) -> Issue:
        if issue_id not in self.state.issues:
            raise ValueError(f"Issue {issue_id} not found.")
        return self.state.issues[issue_id]

    def _get_observation(self) -> Observation:
        summaries = []
        for i_id, issue in self.state.issues.items():
            summaries.append(IssueSummary(
                id=issue.id,
                title=issue.title,
                status=issue.status,
                labels=issue.labels,
                priority=issue.priority,
                assignee=issue.assignee
            ))
            
        task_info = TASKS[self.state.current_task_id]
        
        return Observation(
            inbox=summaries,
            last_action_result=self.state.last_action_result,
            step_count=self.state.step_count,
            max_steps=self.state.max_steps,
            task_description=task_info["description"]
        )
