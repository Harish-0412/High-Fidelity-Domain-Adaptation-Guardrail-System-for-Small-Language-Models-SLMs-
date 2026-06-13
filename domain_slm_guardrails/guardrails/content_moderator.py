from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ContentModerationResult:
    is_safe: bool
    flagged_categories: List[str]
    confidence: float
    explanation: str


class ContentModerator:
    """Basic content moderation guardrail for healthcare domain."""

    def __init__(self):
        # Medical domain specific guardrails
        self.moderation_patterns = {
            "harmful_advice": [
                r"suicide|kill yourself|end your life",
                r"ignore your doctor|stop taking|skip your medication",
                r"don't seek help|avoid medical attention",
            ],
            "personal_info_request": [
                r"ssn|social security|credit card|bank account",
            ],
            "diagnosis_prompt": [
                r"diagnose me|what disease do i have|am i sick",
            ],
        }

    def check(self, text: str, context: Optional[str] = None) -> ContentModerationResult:
        flagged = []
        text_lower = text.lower()

        for category, patterns in self.moderation_patterns.items():
            for pat in patterns:
                # Use word boundaries to avoid partial matches
                if re.search(r"\b(" + pat + r")\b", text_lower, re.IGNORECASE):
                    flagged.append(category)
                    break

        is_safe = len(flagged) == 0
        confidence = 0.85 if flagged else 0.7

        explanation = (
            "No harmful or unsafe content detected"
            if is_safe
            else f"Content flagged in categories: {', '.join(flagged)}"
        )

        return ContentModerationResult(
            is_safe=is_safe,
            flagged_categories=flagged,
            confidence=confidence,
            explanation=explanation,
        )
