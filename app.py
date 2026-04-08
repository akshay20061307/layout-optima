"""
app.py — Layout-Optima FastAPI server + Gradio UI.

Exposes OpenEnv-compliant HTTP endpoints:
    POST /reset          → initial observation
    POST /step           → (obs, reward, done, info)
    GET  /state          → episode metadata
    GET  /tasks          → list of task ids
    GET  /health         → 200 OK

Also serves a Gradio web interface at /web for interactive demos.
"""

import json
import os
from typing import Any, Dict, List, Optional

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import LayoutOptimaEnv
from tasks import TASK_DESCRIPTIONS, TASK_DIFFICULTY, TASKS, get_task_info

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

api = FastAPI(
    title="Layout-Optima OpenEnv",
    description="AI-driven UI/UX layout optimisation RL environment.",
    version="1.0.0",
)

# One global env instance per process (stateless per reset call)
_envs: Dict[str, LayoutOptimaEnv] = {}


class ResetRequest(BaseModel):
    task_id: str = "cta_visibility"
    max_steps: int = 20


class StepRequest(BaseModel):
    task_id: str = "cta_visibility"
    action: Dict[str, Any]


@api.get("/health")
def health():
    return {"status": "ok", "tasks": list(TASKS.keys())}


@api.get("/tasks")
def list_tasks():
    return {
        task_id: {
            "description": TASK_DESCRIPTIONS[task_id],
            "difficulty": TASK_DIFFICULTY[task_id],
        }
        for task_id in TASKS
    }


@api.post("/reset")
def reset(req: ResetRequest):
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{req.task_id}'")
    env = LayoutOptimaEnv(req.task_id, max_steps=req.max_steps)
    obs = env.reset()
    _envs[req.task_id] = env
    return JSONResponse(content=obs)


@api.post("/step")
def step(req: StepRequest):
    env = _envs.get(req.task_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active episode for task '{req.task_id}'. Call /reset first.",
        )
    if env.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset.")
    obs, reward, done, info = env.step(req.action)
    return JSONResponse(
        content={"observation": obs, "reward": reward, "done": done, "info": info}
    )


@api.get("/state")
def state(task_id: str = "cta_visibility"):
    env = _envs.get(task_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active episode for task '{task_id}'. Call /reset first.",
        )
    return JSONResponse(content=env.state())


# ---------------------------------------------------------------------------
# Gradio UI helpers
# ---------------------------------------------------------------------------

TYPE_COLORS = {
    "hero": "#4CAF50", "feature": "#2196F3", "testimonial": "#FF9800",
    "cta": "#F44336", "form": "#9C27B0", "pricing": "#00BCD4",
    "faq": "#607D8B", "blog": "#795548",
}
TYPE_ICONS = {
    "hero": "🏠", "feature": "⭐", "testimonial": "💬",
    "cta": "🔘", "form": "📝", "pricing": "💰", "faq": "❓", "blog": "📰",
}

_ui_env: Optional[LayoutOptimaEnv] = None
_ui_history: List[Dict] = []


def _render_layout(obs: Dict[str, Any]) -> str:
    if not obs or "components" not in obs:
        return "<p>No layout data</p>"
    score = obs.get("score", 0)
    score_color = "#4CAF50" if score >= 1.0 else "#FF9800" if score >= 0.5 else "#F44336"
    parts = [f"""
    <div style="background:#1a1a2e;color:white;padding:15px;border-radius:10px;margin-bottom:15px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h3 style="margin:0;">{obs.get('task_id','').replace('_',' ').title()}</h3>
        <div style="background:{score_color};padding:8px 16px;border-radius:20px;font-weight:bold;">
          Score: {score:.2f}
        </div>
      </div>
      <p style="margin:5px 0 0 0;opacity:0.8;">{obs.get('task_description','')}</p>
    </div>"""]

    for comp in obs["components"]:
        ct = comp.get("type", "unknown")
        color = TYPE_COLORS.get(ct, "#666")
        icon = TYPE_ICONS.get(ct, "📦")
        badges = []
        if comp.get("click_count", 0) > 200 and comp.get("conversion_rate", 1) < 0.10:
            badges.append('<span style="background:#F44336;padding:2px 8px;border-radius:10px;font-size:11px;">⚠️ FRICTION</span>')
        if ct == "testimonial":
            badges.append('<span style="background:#FF9800;padding:2px 8px;border-radius:10px;font-size:11px;">✓ TRUST</span>')
        if ct == "cta":
            badges.append('<span style="background:#2196F3;padding:2px 8px;border-radius:10px;font-size:11px;">🔘 CTA</span>')
        badges_html = " ".join(badges)
        parts.append(f"""
        <div style="background:linear-gradient(135deg,{color}22,{color}11);border-left:4px solid {color};
             padding:12px 15px;margin-bottom:8px;border-radius:0 8px 8px 0;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div><span style="font-size:18px;margin-right:8px;">{icon}</span>
              <strong>{comp.get('label', comp.get('id','?'))}</strong>
              <span style="opacity:0.6;margin-left:8px;">({comp.get('id','')})</span>
            </div>
            <span style="background:{color}44;padding:4px 10px;border-radius:12px;font-size:12px;">
              Index: {comp.get('index','?')}
            </span>
          </div>
          <div style="display:flex;gap:15px;margin-top:8px;font-size:13px;opacity:0.85;">
            <span>⏱ {comp.get('dwell_time',0):.1f}s</span>
            <span>👆 {comp.get('click_count',0)}</span>
            <span>🎯 {comp.get('conversion_rate',0):.1%}</span>
            <span>📜 {comp.get('scroll_depth',0):.0%}</span>
          </div>
          {f'<div style="margin-top:6px;">{badges_html}</div>' if badges else ''}
        </div>""")
    return "\n".join(parts)


def _render_heatmap(obs: Dict[str, Any]) -> str:
    if not obs or "heatmap_summary" not in obs:
        return "<p>No heatmap data</p>"
    hm = obs["heatmap_summary"]
    parts = ["<h4>📊 Heatmap Analysis</h4>"]
    if hm.get("top_dwell_time"):
        parts.append("<p><strong>Top Engagement:</strong></p><ul>")
        for item in hm["top_dwell_time"]:
            parts.append(f'<li>{item["label"]}: {item["dwell_time"]:.1f}s</li>')
        parts.append("</ul>")
    if hm.get("friction_points"):
        parts.append('<p style="color:#F44336;"><strong>⚠️ Friction Points:</strong></p><ul>')
        for item in hm["friction_points"]:
            parts.append(
                f'<li>{item["label"]}: {item["click_count"]} clicks, '
                f'{item["conversion_rate"]:.1%} conv '
                f'(friction: {item["friction_score"]})</li>'
            )
        parts.append("</ul>")
    if hm.get("trust_signals"):
        parts.append('<p style="color:#FF9800;"><strong>✓ Trust Signals:</strong></p><ul>')
        for item in hm["trust_signals"]:
            parts.append(f'<li>{item["label"]} at index {item["index"]}</li>')
        parts.append("</ul>")
    return "\n".join(parts)


def _render_history(history: List[Dict]) -> str:
    if not history:
        return "<p>No actions yet.</p>"
    parts = [
        "<table style='width:100%;border-collapse:collapse;'>",
        "<tr style='background:#333;color:white;'>"
        "<th style='padding:8px;text-align:left;'>Step</th>"
        "<th style='padding:8px;text-align:left;'>Action</th>"
        "<th style='padding:8px;text-align:left;'>Reward</th>"
        "<th style='padding:8px;text-align:left;'>Score</th>"
        "<th style='padding:8px;text-align:left;'>Done</th>"
        "</tr>",
    ]
    for h in history[-10:]:
        act_str = json.dumps(h.get("action", {}), separators=(",", ":"))
        if len(act_str) > 50:
            act_str = act_str[:47] + "..."
        bg = "background:#2a2a2a;" if h["step"] % 2 == 0 else ""
        rwd = h["reward"]
        parts.append(
            f"<tr style='{bg}'>"
            f"<td style='padding:8px;'>{h['step']}</td>"
            f"<td style='padding:8px;font-family:monospace;font-size:12px;'>{act_str}</td>"
            f"<td style='padding:8px;color:{'green' if rwd > 0 else 'red'};'>{rwd:.3f}</td>"
            f"<td style='padding:8px;'>{h['score']:.3f}</td>"
            f"<td style='padding:8px;'>{'✓' if h['done'] else ''}</td>"
            "</tr>"
        )
    parts.append("</table>")
    return "\n".join(parts)


def on_task_select(task_id: str):
    global _ui_env, _ui_history
    _ui_env = LayoutOptimaEnv(task_id, max_steps=20)
    obs = _ui_env.reset()
    _ui_history = []
    info = get_task_info(task_id)
    examples = [
        '{"type": "move_component", "id": "CTA_BUTTON", "new_index": 1}',
        '{"type": "move_component", "id": "SEC_TESTIMONIALS", "new_index": 0}',
        '{"type": "move_component", "id": "FF_TRUST", "new_index": 0}',
    ]
    task_text = (
        f"Task: {task_id}\n"
        f"Difficulty: {info['difficulty']}\n"
        f"Components: {info['num_components']}\n"
        f"Friction Points: {info['num_friction_points']}"
    )
    return (
        _render_layout(obs),
        _render_heatmap(obs),
        _render_history([]),
        gr.update(value=examples[0], choices=examples),
        gr.update(value=task_text),
    )


def on_execute_action(action_str: str):
    global _ui_history
    if _ui_env is None or _ui_env.done:
        obs = _ui_env._observe() if _ui_env else {}
        return _render_layout(obs), _render_history(_ui_history)
    try:
        action = json.loads(action_str)
    except json.JSONDecodeError:
        return "<p style='color:red;'>Invalid JSON</p>", _render_history(_ui_history)
    obs, reward, done, info = _ui_env.step(action)
    _ui_history.append({
        "step": len(_ui_history) + 1,
        "action": action,
        "reward": reward,
        "score": obs["score"],
        "done": done,
    })
    return _render_layout(obs), _render_history(_ui_history)


def on_auto_solve():
    global _ui_history
    if _ui_env is None:
        return "", ""
    from inference import RuleBasedAgent
    agent = RuleBasedAgent(_ui_env.task_id)
    while not _ui_env.done:
        obs = _ui_env._observe()
        actions = agent.act(obs)
        if not actions:
            break
        action = actions[0]
        obs, reward, done, info = _ui_env.step(action)
        _ui_history.append({
            "step": len(_ui_history) + 1,
            "action": action,
            "reward": round(reward, 4),
            "score": round(obs["score"], 4),
            "done": done,
        })
    final_obs = _ui_env._observe()
    return _render_layout(final_obs), _render_history(_ui_history)


def on_reset():
    if _ui_env:
        return on_task_select(_ui_env.task_id)
    return "", "", "", gr.update(), gr.update()


# ---------------------------------------------------------------------------
# Gradio blocks
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Layout-Optima: AI UI/UX Optimisation",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"),
    css="footer{display:none!important}",
) as demo:
    gr.Markdown(
        "# 🎨 Layout-Optima\n"
        "### AI-Driven UI/UX Layout Optimisation — Scaler × Meta × HuggingFace Hackathon\n"
        "Simulate CRO by rearranging webpage components to maximise engagement and conversions."
    )

    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(
                choices=list(TASKS.keys()),
                value="cta_visibility",
                label="Select Task",
                info="Choose a layout optimisation challenge",
            )
            task_info = gr.Textbox(label="Task Info", lines=4, interactive=False)

        with gr.Column(scale=2):
            action_input = gr.Dropdown(
                choices=['{"type": "move_component", "id": "CTA_BUTTON", "new_index": 1}'],
                value='{"type": "move_component", "id": "CTA_BUTTON", "new_index": 1}',
                label="Action (JSON)",
                allow_custom_value=True,
            )
            with gr.Row():
                exec_btn = gr.Button("▶️ Execute Action", variant="primary")
                auto_btn = gr.Button("🤖 Auto-Solve", variant="secondary")
                reset_btn = gr.Button("🔄 Reset", variant="stop")

    with gr.Row():
        with gr.Column(scale=2):
            layout_output = gr.HTML(label="Current Layout", value="<p>Select a task to begin</p>")
        with gr.Column(scale=1):
            heatmap_output = gr.HTML(label="Heatmap Analysis", value="<p>Select a task to begin</p>")

    history_output = gr.HTML(label="Action History", value="<p>No actions taken yet</p>")

    task_dropdown.change(
        fn=on_task_select,
        inputs=[task_dropdown],
        outputs=[layout_output, heatmap_output, history_output, action_input, task_info],
    )
    exec_btn.click(fn=on_execute_action, inputs=[action_input], outputs=[layout_output, history_output])
    auto_btn.click(fn=on_auto_solve, inputs=[], outputs=[layout_output, history_output])
    reset_btn.click(fn=on_reset, inputs=[], outputs=[layout_output, heatmap_output, history_output, action_input, task_info])
    demo.load(
        fn=lambda: on_task_select("cta_visibility"),
        inputs=[],
        outputs=[layout_output, heatmap_output, history_output, action_input, task_info],
    )

# ---------------------------------------------------------------------------
# Mount Gradio onto FastAPI and run
# ---------------------------------------------------------------------------

app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)