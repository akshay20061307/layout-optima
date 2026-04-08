"""
env.py — Layout-Optima RL environment.
Implements reset() / step() / state() / grade() per OpenEnv spec.
"""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

from tasks import (
    CLICK_THRESHOLD, CONV_THRESHOLD, TASK_DESCRIPTIONS, TASKS, Component,
)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class LayoutOptimaEnv:
    """
    Single-episode environment for UI layout optimisation.

    Follows OpenEnv spec:
        obs          = env.reset()
        obs, r, done, info = env.step(action)
        state_info   = env.state()
        score        = env.grade()
    """

    VALID_ACTIONS = {"move_component", "change_color", "remove_component"}
    STEP_BONUS = 0.05
    INVALID_PENALTY = -0.02
    NOOP_PENALTY = -0.01

    def __init__(self, task_id: str, max_steps: int = 20):
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Choose from: {list(TASKS)}"
            )
        self.task_id = task_id
        self.max_steps = max_steps
        self._master: List[Component] = TASKS[task_id]
        self.components: List[Component] = []
        self.step_count: int = 0
        self.done: bool = False
        self.history: List[Dict] = []
        self._prev_score: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        self.components = copy.deepcopy(self._master)
        self.step_count = 0
        self.done = False
        self.history = []
        self._prev_score = self.grade()
        return self._observe()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode is over. Call reset() to start a new one.")

        reward = 0.0
        info: Dict[str, Any] = {"action_type": action.get("type", "unknown")}
        action_type = action.get("type")
        comp_id = action.get("id")

        if action_type not in self.VALID_ACTIONS:
            reward = self.INVALID_PENALTY
            info["error"] = f"Invalid action type '{action_type}'"
        elif comp_id is None:
            reward = self.INVALID_PENALTY
            info["error"] = "Missing 'id' field in action"
        else:
            component = self._get_component(comp_id)
            if component is None:
                reward = self.INVALID_PENALTY
                info["error"] = f"Component '{comp_id}' not found"
            elif not component.visible:
                reward = self.INVALID_PENALTY
                info["error"] = f"Component '{comp_id}' is not visible"
            else:
                reward, info = self._dispatch(action_type, component, action)

        self.step_count += 1
        self.history.append({
            "step": self.step_count,
            "action": action,
            "reward": reward,
            "info": info,
            "score_after": self.grade(),
        })

        terminal_score = self.grade()
        if terminal_score >= 1.0:
            self.done = True
            info["terminated"] = "task_solved"
            reward += 0.5
        elif self.step_count >= self.max_steps:
            self.done = True
            info["terminated"] = "max_steps_reached"

        self._prev_score = terminal_score
        return self._observe(), reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Return current episode metadata (OpenEnv state() API)."""
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
            "score": round(self.grade(), 4),
            "history_len": len(self.history),
        }

    # ------------------------------------------------------------------
    # Graders
    # ------------------------------------------------------------------

    def grade(self) -> float:
        if self.task_id == "cta_visibility":
            return self._grade_cta_visibility()
        if self.task_id == "content_prioritize":
            return self._grade_content_prioritize()
        if self.task_id == "form_friction":
            return self._grade_form_friction()
        return 0.0

    def _grade_cta_visibility(self) -> float:
        cta = self._get_component("CTA_BUTTON")
        if cta is None or not cta.visible:
            return 0.0
        n = len(self.components)
        if n == 0:
            return 0.0
        if cta.index < 2:
            return 1.0
        return max(0.0, 1.0 - (cta.index - 1) / (n - 1))

    def _grade_content_prioritize(self) -> float:
        visible = [c for c in self.components if c.visible]
        if len(visible) < 2:
            return 0.0
        sorted_by_dwell = sorted(visible, key=lambda c: c.dwell_time, reverse=True)
        dwell_rank = {c.id: rank for rank, c in enumerate(sorted_by_dwell)}
        indices = [c.index for c in visible]
        dwell_ranks = [dwell_rank[c.id] for c in visible]
        return max(0.0, _pearson(indices, dwell_ranks))

    def _grade_form_friction(self) -> float:
        visible = sorted(
            [c for c in self.components if c.visible], key=lambda c: c.index
        )
        friction_ids = {
            c.id for c in visible
            if c.click_count > CLICK_THRESHOLD and c.conversion_rate < CONV_THRESHOLD
        }
        if not friction_ids:
            return 1.0
        shielded = 0
        total = len(friction_ids)
        last_was_trust = False
        for c in visible:
            if c.id in friction_ids:
                if last_was_trust:
                    shielded += 1
                last_was_trust = False
            else:
                last_was_trust = (c.type == "testimonial")
        return shielded / total if total > 0 else 1.0

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, action_type: str, component: Component, action: Dict
    ) -> Tuple[float, Dict]:
        pre_score = self.grade()

        if action_type == "move_component":
            new_index = action.get("new_index")
            if new_index is None or not isinstance(new_index, int):
                return self.INVALID_PENALTY, {"error": "move_component requires integer 'new_index'"}
            new_index = max(0, min(new_index, len(self.components) - 1))
            if component.index == new_index:
                return self.NOOP_PENALTY, {"info": "no_position_change"}
            self._move(component, new_index)

        elif action_type == "change_color":
            hex_color = action.get("hex", "#FFFFFF")
            if not self._valid_hex(hex_color):
                return self.INVALID_PENALTY, {"error": f"Invalid hex '{hex_color}'"}
            component.color = hex_color

        elif action_type == "remove_component":
            component.visible = False
            self._reindex()

        post_score = self.grade()
        delta = post_score - pre_score
        reward = delta + (self.STEP_BONUS if delta > 0 else 0.0)
        return reward, {"delta_score": round(delta, 4), "post_score": round(post_score, 4)}

    def _valid_hex(self, h: str) -> bool:
        if not h or len(h) != 7 or h[0] != "#":
            return False
        try:
            int(h[1:], 16)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Observation / internal helpers
    # ------------------------------------------------------------------

    def _observe(self) -> Dict[str, Any]:
        visible = sorted(
            [c for c in self.components if c.visible], key=lambda c: c.index
        )
        return {
            "task_id": self.task_id,
            "task_description": TASK_DESCRIPTIONS.get(self.task_id, ""),
            "step": self.step_count,
            "max_steps": self.max_steps,
            "score": round(self.grade(), 4),
            "components": [c.to_obs_dict() for c in visible],
            "heatmap_summary": self._heatmap_summary(visible),
            "available_actions": list(self.VALID_ACTIONS),
        }

    def _heatmap_summary(self, visible: List[Component]) -> Dict[str, Any]:
        friction = [
            {
                "id": c.id,
                "label": c.label,
                "click_count": c.click_count,
                "conversion_rate": c.conversion_rate,
                "friction_score": round(c.click_count * (1 - c.conversion_rate), 2),
            }
            for c in visible
            if c.click_count > CLICK_THRESHOLD and c.conversion_rate < CONV_THRESHOLD
        ]
        top_dwell = sorted(visible, key=lambda c: c.dwell_time, reverse=True)[:3]
        return {
            "top_dwell_time": [
                {"id": c.id, "label": c.label, "dwell_time": c.dwell_time}
                for c in top_dwell
            ],
            "friction_points": friction,
            "trust_signals": [
                {"id": c.id, "label": c.label, "index": c.index}
                for c in visible if c.type == "testimonial"
            ],
            "cta_components": [
                {"id": c.id, "label": c.label, "index": c.index}
                for c in visible if c.type == "cta"
            ],
        }

    def _get_component(self, comp_id: str) -> Optional[Component]:
        return next((c for c in self.components if c.id == comp_id), None)

    def _move(self, component: Component, new_index: int) -> None:
        old_index = component.index
        if old_index == new_index:
            return
        if new_index < old_index:
            for c in self.components:
                if c.id != component.id and new_index <= c.index < old_index:
                    c.index += 1
        else:
            for c in self.components:
                if c.id != component.id and old_index < c.index <= new_index:
                    c.index -= 1
        component.index = new_index

    def _reindex(self) -> None:
        visible = sorted(
            [c for c in self.components if c.visible], key=lambda c: c.index
        )
        for i, c in enumerate(visible):
            c.index = i

    def get_layout_summary(self) -> str:
        visible = sorted(
            [c for c in self.components if c.visible], key=lambda c: c.index
        )
        lines = [f"Layout for '{self.task_id}' (score: {self.grade():.2f}):"]
        for c in visible:
            tags = ""
            if c.is_friction:
                tags += " ⚠️ FRICTION"
            if c.is_trust_signal:
                tags += " ✓ TRUST"
            if c.is_cta:
                tags += " 🔘 CTA"
            lines.append(f"  [{c.index}] {c.id}: {c.label}{tags}")
        return "\n".join(lines)