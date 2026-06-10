# Medical Prescription Groundedness Comparison Report

## Overview
This report compares baseline supervised fine-tuning (SFT) outputs against DPO-aligned policy outputs for the medical prescription domain. The goal is to quantify how preference alignment improves citation grounding, reduces hallucination, preserves conciseness, and increases safe fallback behavior.

## Summary
- Baseline model: SFT policy
- Policy model: DPO-aligned adapter
- Evaluation focus: citation density, grounding score, factual consistency, hallucination penalty, conciseness, fallback behavior

## Key Metrics
| Metric | Baseline | DPO Policy | Delta |
|---|---|---|---|
| Average grounding score | TBD | TBD | TBD |
| Average citation density | TBD | TBD | TBD |
| Average hallucination penalty | TBD | TBD | TBD |
| Average conciseness | TBD | TBD | TBD |
| Fallback rate | TBD | TBD | TBD |

## Findings
- DPO alignment should prioritize factual, citation-backed outputs over vague or hallucinated alternatives.
- The comparator is designed to identify improvements in both raw answer quality and guardrail compliance.
- A strong DPO policy will show higher grounding scores, lower hallucination penalties, and a stable or improved conciseness metric.

## Failure Modes
- `weakly_cited` rejections reveal when the model still prefers low-evidence wording.
- `hallucinated` comparisons expose unsupported claims that should be down-ranked.
- `incomplete` responses demonstrate whether the model can avoid truncating important conclusions.
- `overly_verbose` responses measure whether the policy has learned to be concise while grounded.

## Recommendations
- Continue tuning `beta` and learning rate until the policy consistently prefers cited answers.
- Validate that DPO training does not degrade baseline SFT domain knowledge by checking grounding scores on held-out prompts.
- Use the comparator output to identify prompts where policy outputs still fall back too often or remain ungrounded.

## Next Steps
1. Export comparator results to CSV and JSON for dashboard ingestion.
2. Add charts for grounding score delta and fallback rate improvements.
3. Iterate on rejected-answer generation strategies using retrieval evidence and template-based perturbations.
