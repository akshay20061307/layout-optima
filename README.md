---
title: Layout-Optima
emoji: 🎨
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🎨 Layout-Optima

**AI-Driven UI/UX Layout Optimisation** — Scaler × Meta × HuggingFace Hackathon submission.

Simulates CRO (Conversion Rate Optimisation) by letting an agent rearrange webpage
components to maximise engagement and conversion scores across 3 tasks of increasing
difficulty.

---

## Tasks

| Task | Difficulty | Goal | Score Metric |
|---|---|---|---|
| `cta_visibility` | Easy | Move CTA button above the fold (index < 2) | Linear proximity |
| `content_prioritize` | Medium | Sort sections by descending dwell time | Pearson correlation |
| `form_friction` | Hard | Place trust signals before friction components | Shielded fraction |

---

## Observation Space

```json
{
  "task_id": "cta_visibility",
  "task_description": "Move the CTA button above the fold (index < 2)...",
  "step": 0,
  "max_steps": 20,
  "score": 0.2,
  "components": [
    {
      "id": "CTA_BUTTON",
      "type": "cta",
      "label": "Sign Up Free",
      "index": 5,
      "dwell_time": 1.1,
      "click_count": 22,
      "conversion_rate": 0.41,
      "scroll_depth": 0.15
    }
  ],
  "heatmap_summary": {
    "top_dwell_time": [...],
    "friction_points": [...],
    "trust_signals": [...],
    "cta_components": [...]
  },
  "available_actions": ["move_component", "change_color", "remove_component"]
}
```

## Action Space

```json
{"type": "move_component",  "id": "<component_id>", "new_index": 1}
{"type": "change_color",    "id": "<component_id>", "hex": "#FF5733"}
{"type": "remove_component","id": "<component_id>"}
```

## Reward

| Event | Reward |
|---|---|
| Score improvement (delta) | `delta + 0.05` |
| No improvement | `delta` (≤ 0) |
| Task solved (score ≥ 1.0) | `+0.5` bonus |
| Invalid action | `-0.02` |
| No-op | `-0.01` |

---

## HTTP API (OpenEnv spec)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns 200 |
| `GET` | `/tasks` | List all task ids |
| `POST` | `/reset` | Start new episode `{"task_id":"cta_visibility"}` |
| `POST` | `/step` | Execute action `{"task_id":"...", "action":{...}}` |
| `GET` | `/state?task_id=...` | Episode metadata |

---

## Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Model identifier for inference |
| `HF_TOKEN` | Hugging Face / API key |

---

## Run Locally

$env:ENV_BASE_URL = "https://tanishagupta30-layout-optima.hf.space"
>> python inference.py

## File Structure

```
├── app.py            # FastAPI + Gradio server (OpenEnv endpoints + web UI)
├── env.py            # LayoutOptimaEnv — reset/step/state/grade logic
├── inference.py      # Inference runner with [START]/[STEP]/[END] logs
├── tasks.py          # Component dataclass + 3 task definitions
├── openenv.yaml      # OpenEnv spec manifest
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
└── README.md         # This file
```