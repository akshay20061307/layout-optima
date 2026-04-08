"""
tasks.py — Layout-Optima task definitions.
Defines Component dataclass, 3 tasks (easy/medium/hard), and task registry.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Friction detection thresholds
# ---------------------------------------------------------------------------
CLICK_THRESHOLD = 200
CONV_THRESHOLD = 0.10


@dataclass
class Component:
    """Represents a single UI component with engagement metrics."""
    id: str
    type: str  # hero / cta / feature / testimonial / form / pricing / faq / blog
    label: str
    index: int  # current position (0 = top)
    dwell_time: float  # avg seconds users spend on this component
    click_count: int  # total clicks recorded in session logs
    conversion_rate: float  # 0.0–1.0
    scroll_depth: float  # 0.0–1.0
    color: str = "#FFFFFF"
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    optimal_index: Optional[int] = None

    def to_obs_dict(self) -> Dict[str, Any]:
        """Observation dict exposed to agents — no internal fields."""
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "index": self.index,
            "dwell_time": self.dwell_time,
            "click_count": self.click_count,
            "conversion_rate": self.conversion_rate,
            "scroll_depth": self.scroll_depth,
        }

    @property
    def is_friction(self) -> bool:
        return self.click_count > CLICK_THRESHOLD and self.conversion_rate < CONV_THRESHOLD

    @property
    def is_trust_signal(self) -> bool:
        return self.type == "testimonial"

    @property
    def is_cta(self) -> bool:
        return self.type == "cta"


# ---------------------------------------------------------------------------
# Task 1 — cta_visibility (Easy)
# Fix: Place CTA_BUTTON at index < 2
# ---------------------------------------------------------------------------
CTA_VISIBILITY_LAYOUT: List[Component] = [
    Component(id="HERO_BANNER", type="hero", label="Hero Banner", index=0, dwell_time=8.2, click_count=120, conversion_rate=0.05, scroll_depth=0.95, optimal_index=0),
    Component(id="FEATURE_LIST", type="feature", label="Feature Highlights", index=1, dwell_time=12.4, click_count=340, conversion_rate=0.08, scroll_depth=0.70, optimal_index=1),
    Component(id="PRICING_TABLE", type="pricing", label="Pricing Plans", index=2, dwell_time=6.1, click_count=210, conversion_rate=0.12, scroll_depth=0.55, optimal_index=3),
    Component(id="TESTIMONIALS", type="testimonial", label="Customer Testimonials", index=3, dwell_time=4.8, click_count=95, conversion_rate=0.03, scroll_depth=0.40, optimal_index=4),
    Component(id="FAQ_SECTION", type="faq", label="FAQ", index=4, dwell_time=3.3, click_count=60, conversion_rate=0.02, scroll_depth=0.30, optimal_index=5),
    Component(id="CTA_BUTTON", type="cta", label="Sign Up Free", index=5, dwell_time=1.1, click_count=22, conversion_rate=0.41, scroll_depth=0.15, optimal_index=2),
]

# ---------------------------------------------------------------------------
# Task 2 — content_prioritize (Medium)
# Fix: Sort sections by descending dwell_time (Pearson correlation)
# ---------------------------------------------------------------------------
CONTENT_PRIORITIZE_LAYOUT: List[Component] = [
    Component(id="SEC_FAQ", type="faq", label="FAQ", index=0, dwell_time=3.1, click_count=55, conversion_rate=0.02, scroll_depth=0.28, optimal_index=6),
    Component(id="SEC_HERO", type="hero", label="Hero", index=1, dwell_time=7.5, click_count=130, conversion_rate=0.06, scroll_depth=0.90, optimal_index=4),
    Component(id="SEC_TESTIMONIALS", type="testimonial", label="Testimonials", index=2, dwell_time=14.2, click_count=420, conversion_rate=0.18, scroll_depth=0.80, optimal_index=0),
    Component(id="SEC_PRICING", type="pricing", label="Pricing", index=3, dwell_time=9.8, click_count=310, conversion_rate=0.22, scroll_depth=0.65, optimal_index=2),
    Component(id="SEC_FEATURES", type="feature", label="Feature Highlights", index=4, dwell_time=11.6, click_count=280, conversion_rate=0.14, scroll_depth=0.72, optimal_index=1),
    Component(id="SEC_BLOG", type="blog", label="Blog Preview", index=5, dwell_time=2.4, click_count=40, conversion_rate=0.01, scroll_depth=0.20, optimal_index=7),
    Component(id="SEC_CTA", type="cta", label="CTA Banner", index=6, dwell_time=5.5, click_count=180, conversion_rate=0.35, scroll_depth=0.50, optimal_index=3),
    Component(id="SEC_PARTNERS", type="testimonial", label="Partner Logos", index=7, dwell_time=1.8, click_count=25, conversion_rate=0.01, scroll_depth=0.15, optimal_index=5),
]

# ---------------------------------------------------------------------------
# Task 3 — form_friction (Hard)
# Fix: Place trust signals immediately before EACH friction component.
#
# Friction components (click_count > 200 AND conversion_rate < 0.10):
#   FF_CHECKOUT  (clicks=520, conv=0.06) ✓
#   FF_UPSELL    (clicks=380, conv=0.04) ✓
#   FF_NEWSLETTER(clicks=260, conv=0.08) ✓
#
# With only 1 trust signal the grader can shield at most 1/3 friction points
# (score=0.333), making score>=1.0 unreachable → the episode can NEVER be
# solved, so success=false always.  Fix: add a second trust signal
# (FF_TRUST2) so the agent can shield all 3 friction points with 2 moves,
# achieving score=1.0 in a solvable number of steps.
#
# Winning layout: FF_TRUST(before FF_CHECKOUT) + FF_TRUST2(before FF_UPSELL)
#   → shielded=2; FF_NEWSLETTER is the 3rd friction point still needing cover.
#
# To make it fully solvable (score=1.0): we change FF_NEWSLETTER's
# conversion_rate to 0.12 (above CONV_THRESHOLD) so it is no longer a
# friction point, leaving only 2 friction points for 2 trust signals → 2/2=1.0
# ---------------------------------------------------------------------------
FORM_FRICTION_LAYOUT: List[Component] = [
    Component(id="FF_HERO",       type="hero",        label="Hero Banner",       index=0, dwell_time=7.0, click_count=100, conversion_rate=0.07, scroll_depth=0.88, optimal_index=4),
    Component(id="FF_CHECKOUT",   type="form",        label="Checkout Form",     index=1, dwell_time=9.3, click_count=520, conversion_rate=0.06, scroll_depth=0.60, metadata={"fields": ["name","email","card","address"]}, optimal_index=2),
    Component(id="FF_UPSELL",     type="cta",         label="Premium Upsell",    index=2, dwell_time=4.5, click_count=380, conversion_rate=0.04, scroll_depth=0.42, optimal_index=3),
    Component(id="FF_TRUST",      type="testimonial", label="★ Verified Reviews", index=3, dwell_time=6.2, click_count=140, conversion_rate=0.02, scroll_depth=0.35, optimal_index=1),
    # FF_TRUST2: second trust signal added to make the task solvable (score=1.0 reachable)
    Component(id="FF_TRUST2",     type="testimonial", label="★ Expert Endorsements", index=4, dwell_time=5.1, click_count=110, conversion_rate=0.02, scroll_depth=0.30, optimal_index=3),
    # FF_NEWSLETTER conv_rate raised to 0.12 so it is no longer a friction point,
    # leaving exactly 2 friction points (FF_CHECKOUT, FF_UPSELL) for 2 trust signals.
    Component(id="FF_NEWSLETTER", type="form",        label="Newsletter Signup", index=5, dwell_time=2.8, click_count=260, conversion_rate=0.12, scroll_depth=0.25, optimal_index=5),
    Component(id="FF_FOOTER_CTA", type="cta",         label="Get Started Today", index=6, dwell_time=1.5, click_count=90,  conversion_rate=0.38, scroll_depth=0.15, optimal_index=6),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TASKS: Dict[str, List[Component]] = {
    "cta_visibility":    CTA_VISIBILITY_LAYOUT,
    "content_prioritize": CONTENT_PRIORITIZE_LAYOUT,
    "form_friction":     FORM_FRICTION_LAYOUT,
}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "cta_visibility":    "Move the CTA button above the fold (index < 2) to increase visibility.",
    "content_prioritize": "Sort sections by descending dwell time to match user engagement patterns.",
    "form_friction":     "Place trust signals immediately before friction components to reduce checkout abandonment.",
}

TASK_DIFFICULTY: Dict[str, str] = {
    "cta_visibility":    "Easy",
    "content_prioritize": "Medium",
    "form_friction":     "Hard",
}


def get_task_info(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}")
    components = TASKS[task_id]
    return {
        "task_id":            task_id,
        "description":        TASK_DESCRIPTIONS[task_id],
        "difficulty":         TASK_DIFFICULTY[task_id],
        "num_components":     len(components),
        "num_friction_points": len([c for c in components if c.is_friction]),
        "num_trust_signals":  len([c for c in components if c.is_trust_signal]),
        "num_ctas":           len([c for c in components if c.is_cta]),
        "friction_ids":       [c.id for c in components if c.is_friction],
        "trust_ids":          [c.id for c in components if c.is_trust_signal],
        "cta_ids":            [c.id for c in components if c.is_cta],
    }