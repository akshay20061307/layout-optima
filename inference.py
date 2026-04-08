"""
inference.py — Layout-Optima baseline inference script.

Logs strictly follow the required format:
  [START] task=<id> env=<n> model=<model>
  [STEP]  step=<n> action=<json> reward=<f> done=<bool> error=<msg|null>
  [END]   success=<bool> steps=<n> score=<f> rewards=<r1,r2,...>

Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN env vars.
Runtime < 20 min, runs on vcpu=2 / 8GB RAM.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables (mandatory per hackathon spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MAX_STEPS    = int(os.environ.get("MAX_STEPS", "20"))

# Correct task IDs that exist in the environment
TASKS = ["cta_visibility", "content_prioritize", "form_friction"]

# ---------------------------------------------------------------------------
# OpenAI client (mandatory per hackathon spec)
# ---------------------------------------------------------------------------
def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN if HF_TOKEN else "not-needed",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_comp(obs: Dict, comp_id: str) -> Optional[Dict]:
    return next((c for c in obs.get("components", []) if c["id"] == comp_id), None)


def sorted_comps(obs: Dict) -> List[Dict]:
    return sorted(obs.get("components", []), key=lambda c: c["index"])


def friction_comps(comps: List[Dict]) -> List[Dict]:
    """click_count > 200 AND conversion_rate < 0.10"""
    return [
        c for c in comps
        if c.get("click_count", 0) > 200 and c.get("conversion_rate", 1.0) < 0.10
    ]


def trust_comps(comps: List[Dict]) -> List[Dict]:
    return [c for c in comps if c.get("type") == "testimonial"]


# ---------------------------------------------------------------------------
# Rule-based fallback agent
# ---------------------------------------------------------------------------
class RuleBasedAgent:
    """
    Deterministic baseline — task-specific strategy.

    cta_visibility
        Move the first CTA-type component to index 1.  Done in 1 step.

    content_prioritize
        Build a queue sorted by descending dwell_time and execute moves
        one per step.

    form_friction
        The live env has 3 friction components and 1 trust signal.
        Max achievable score = 1/3 ≈ 0.333.

        Each step:
          1. Find all unshielded friction components (no testimonial
             immediately before them).
          2. Find a free trust signal (not already shielding something).
          3. Move that trust signal to (friction_index - 1).
          4. If nothing useful can be done, emit a safe same-position
             move using a component that ACTUALLY EXISTS in this task
             (not CTA_BUTTON, which doesn't exist in form_friction).
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._content_queue: List[Dict[str, Any]] = []
        self._content_built = False

    def act(self, obs: Dict) -> Dict[str, Any]:
        if self.task_id == "cta_visibility":
            return self._act_cta(obs)
        if self.task_id == "content_prioritize":
            return self._act_content(obs)
        if self.task_id == "form_friction":
            return self._act_friction(obs)
        return self._safe_noop(obs)

    # ------------------------------------------------------------------
    def _act_cta(self, obs: Dict) -> Dict[str, Any]:
        for c in sorted_comps(obs):
            if c["type"] == "cta" and c["index"] > 1:
                return {"type": "move_component", "id": c["id"], "new_index": 1}
        return self._safe_noop(obs)

    # ------------------------------------------------------------------
    def _act_content(self, obs: Dict) -> Dict[str, Any]:
        if not self._content_built:
            comps = sorted(
                obs.get("components", []),
                key=lambda c: c["dwell_time"],
                reverse=True,
            )
            self._content_queue = [
                {"type": "move_component", "id": c["id"], "new_index": i}
                for i, c in enumerate(comps)
                if c["index"] != i
            ]
            self._content_built = True

        while self._content_queue:
            action = self._content_queue.pop(0)
            comp = get_comp(obs, action["id"])
            if comp and comp["index"] != action["new_index"]:
                return action

        return self._safe_noop(obs)

    # ------------------------------------------------------------------
    def _act_friction(self, obs: Dict) -> Dict[str, Any]:
        comps  = sorted_comps(obs)
        trusts = trust_comps(comps)
        fricts = friction_comps(comps)

        if not trusts or not fricts:
            return self._safe_noop(obs)

        index_of = {c["id"]: c["index"] for c in comps}

        # Which friction components are already shielded?
        shielded_ids = set()
        for fc in fricts:
            fi   = index_of[fc["id"]]
            prev = next((c for c in comps if c["index"] == fi - 1), None)
            if prev and prev["type"] == "testimonial":
                shielded_ids.add(fc["id"])

        # Which trust signals are already shielding a friction component?
        shielding_trust_ids = set()
        fric_id_set = {fc["id"] for fc in fricts}
        for tc in trusts:
            ti  = index_of[tc["id"]]
            nxt = next((c for c in comps if c["index"] == ti + 1), None)
            if nxt and nxt["id"] in fric_id_set:
                shielding_trust_ids.add(tc["id"])

        # First unshielded friction component
        unshielded = [fc for fc in fricts if fc["id"] not in shielded_ids]
        if not unshielded:
            return self._safe_noop(obs)

        target_friction = unshielded[0]
        fi = index_of[target_friction["id"]]

        # Free trust signals (not already shielding something else)
        free_trusts = [tc for tc in trusts if tc["id"] not in shielding_trust_ids]
        if not free_trusts:
            free_trusts = trusts  # last resort: re-purpose any trust signal

        trust = free_trusts[0]
        ti    = index_of[trust["id"]]
        desired = max(0, fi - 1)

        if ti == desired:
            # Already in the right place — safe noop
            return self._safe_noop(obs)

        return {"type": "move_component", "id": trust["id"], "new_index": desired}

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_noop(obs: Dict) -> Dict[str, Any]:
        """
        Move the first available component to its own current index.
        This triggers NOOP_PENALTY (-0.01) rather than INVALID_PENALTY
        (-0.02), and never references a component id that doesn't exist
        in the current task.
        """
        comps = obs.get("components", [])
        if comps:
            c = comps[0]
            return {"type": "move_component", "id": c["id"], "new_index": c["index"]}
        return {"type": "move_component", "id": "CTA_BUTTON", "new_index": 1}


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert UI/UX conversion-rate optimisation agent.
You receive the current webpage layout as JSON and output ONE action to improve the score.

Action format (JSON only, no markdown):
  {"type": "move_component", "id": "<component_id>", "new_index": <integer>}

Strategy by task:
  cta_visibility    — move the CTA component (type=cta) to index 1.
  content_prioritize — sort components by descending dwell_time.
  form_friction     — place testimonial components immediately before components
                      with click_count > 200 AND conversion_rate < 0.10.
"""

def llm_act(client: OpenAI, obs: Dict) -> Dict:
    prompt = (
        f"Current layout:\n{json.dumps(obs, indent=2)}\n\n"
        "Output the single best action as JSON:"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=128,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return RuleBasedAgent._safe_noop(obs)


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> Dict:
    r = httpx.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "max_steps": MAX_STEPS},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(task_id: str, action: Dict) -> Dict:
    r = httpx.post(
        f"{ENV_BASE_URL}/step",
        json={"task_id": task_id, "action": action},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Per-task success thresholds
# ---------------------------------------------------------------------------
# The live form_friction env has 3 friction points but only 1 trust signal,
# so the theoretical maximum score = 1/3 ≈ 0.333.  We treat achieving that
# maximum as a pass rather than penalising an env-level constraint.
TASK_SUCCESS_THRESHOLD = {
    "cta_visibility":     1.0,
    "content_prioritize": 1.0,
    "form_friction":      0.333,
}


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------
def run_episode(task_id: str, use_llm: bool) -> Dict:
    client    = get_llm_client() if use_llm else None
    agent     = None if use_llm else RuleBasedAgent(task_id)
    threshold = TASK_SUCCESS_THRESHOLD.get(task_id, 0.8)

    model_label = MODEL_NAME if use_llm else "rule_based"
    print(f"[START] task={task_id} env=layout-optima model={model_label}", flush=True)

    obs         = env_reset(task_id)
    rewards: List[float] = []
    step        = 0
    done        = False
    final_score = obs.get("score", 0.0)
    action: Dict = RuleBasedAgent._safe_noop(obs)

    while not done and step < MAX_STEPS:
        step     += 1
        error_msg = "null"
        try:
            action = llm_act(client, obs) if use_llm else agent.act(obs)

            result      = env_step(task_id, action)
            obs         = result["observation"]
            reward      = float(result["reward"])
            done        = bool(result["done"])
            final_score = obs.get("score", final_score)

        except Exception as e:
            reward    = 0.0
            done      = True
            error_msg = str(e).replace("\n", " ")

        rewards.append(reward)
        action_json = json.dumps(action, separators=(",", ":"))
        print(
            f"[STEP] step={step} action={action_json} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_msg}",
            flush=True,
        )

    rewards_str = ",".join(f"{r:.3f}" for r in rewards)
    success     = final_score >= threshold

    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"score={final_score:.3f} rewards={rewards_str}",
        flush=True,
    )

    return {"task_id": task_id, "score": final_score, "steps": step, "success": success}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    use_llm = bool(HF_TOKEN and HF_TOKEN.strip())

    results = []
    for task_id in TASKS:
        try:
            result = run_episode(task_id, use_llm=use_llm)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] task={task_id} error={e}", flush=True, file=sys.stderr)

    passed  = sum(1 for r in results if r["success"])
    total   = len(results)
    overall = sum(r["score"] for r in results) / total if total else 0.0

    print("\n====================================================", file=sys.stderr)
    print("  LAYOUT-OPTIMA — FINAL RESULTS", file=sys.stderr)
    print("====================================================", file=sys.stderr)
    print(f"  {'TASK':<22} {'SCORE':>7}  {'STEPS':>5}  STATUS", file=sys.stderr)
    print(f"  {'-'*22} {'-'*7}  {'-'*5}  {'-'*7}", file=sys.stderr)
    for r in results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(
            f"  {r['task_id']:<22} {r['score']:>7.3f}  {r['steps']:>5}  {status}",
            file=sys.stderr,
        )
    print("====================================================", file=sys.stderr)
    print(f"  Tasks passed:          {passed}/{total}", file=sys.stderr)
    print(f"  Overall score:         {overall:.3f}", file=sys.stderr)
    print("====================================================", file=sys.stderr)


if __name__ == "__main__":
    main()