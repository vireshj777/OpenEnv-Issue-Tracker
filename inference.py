import os
import json
import logging
from openai import OpenAI
from env import IssueTriageEnv

def run_inference():
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4-turbo-preview")
    hf_token = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

    client = OpenAI(api_key=hf_token, base_url=api_base_url)
    
    # Run locally against our Environment class for the inference script
    env = IssueTriageEnv()
    
    tasks_to_run = ["easy", "medium", "hard"]
    
    for task in tasks_to_run:
        print(f"[START] Task: {task}")
        obs = env.reset(task)
        
        system_prompt = (
            "You are an autonomous agent performing issue triage in an IT tracker system. "
            "Your available actions are represented as JSON objects. Choose ONE action per turn.\n"
            "Action Types:\n"
            '1. {"action_type": "view_issue", "issue_id": "<id>"}\n'
            '2. {"action_type": "add_label", "issue_id": "<id>", "label": "<string>"}\n'
            '3. {"action_type": "set_priority", "issue_id": "<id>", "priority": "low|medium|high|critical"}\n'
            '4. {"action_type": "assign_issue", "issue_id": "<id>", "assignee": "<string>"}\n'
            '5. {"action_type": "close_issue", "issue_id": "<id>", "reason": "completed|duplicate|wont_fix", "duplicate_of": "<id or null>"}\n'
            '6. {"action_type": "submit"}\n\n'
            "Always output a single syntactically valid JSON object representing your chosen action. Provide NO other text, markdown blocks or explanations."
        )

        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        total_reward = 0.0
        done = False
        
        while not done:
            user_msg = f"Current Observation: {obs.model_dump_json()}"
            messages.append({"role": "user", "content": user_msg})
            
            try:
                # Call LLM
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0
                )
                raw_action_str = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[ERROR] LLM API call failed: {e}")
                raw_action_str = '{"action_type": "submit"}'
            
            # Simple cleanse in case model wrapped in markdown
            if raw_action_str.startswith("```json"):
                raw_action_str = raw_action_str[7:-3].strip()
            elif raw_action_str.startswith("```"):
                raw_action_str = raw_action_str[3:-3].strip()
                
            try:
                action_dict = json.loads(raw_action_str)
                # Convert to our pydantic union action using the exact same logic as app.py
                from env import Action
                from models import ViewIssueAction, AddLabelAction, SetPriorityAction, AssignIssueAction, CloseIssueAction, SubmitAction
                
                at = action_dict.get("action_type")
                if at == "view_issue": act = ViewIssueAction(**action_dict)
                elif at == "add_label": act = AddLabelAction(**action_dict)
                elif at == "set_priority": act = SetPriorityAction(**action_dict)
                elif at == "assign_issue": act = AssignIssueAction(**action_dict)
                elif at == "close_issue": act = CloseIssueAction(**action_dict)
                elif at == "submit": act = SubmitAction(**action_dict)
                else: raise ValueError(f"Unknown type {at}")
                
                obs, reward, done, info = env.step(act)
                
            except Exception as e:
                # Malformed action penalty
                action_dict = {"raw": raw_action_str, "error": str(e)}
                from models import Reward
                reward = Reward(value=-0.1, reason=f"Malformed action: {e}")
                obs.last_action_result = f"Error interpreting action: {e}"
                # Force loop to increment but don't call env.step unless we have to, actually we just continue
                env.state.step_count += 1
                if env.state.step_count >= env.state.max_steps:
                    done = True
                    info = {"current_score": env.step(SubmitAction(action_type="submit"))[3].get("current_score", 0)}

            total_reward += reward.value
            
            print(f"[STEP] Action: {json.dumps(action_dict)} | Observation: {obs.model_dump_json()} | Reward: {reward.model_dump_json()}")
            
            messages.append({"role": "assistant", "content": raw_action_str})
            
        print(f"[END] Total Reward: {total_reward} | Done: {done}\n")

if __name__ == "__main__":
    run_inference()
