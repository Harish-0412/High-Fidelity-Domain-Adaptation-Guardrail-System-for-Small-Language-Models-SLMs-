from __future__ import annotations

from typing import Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Base class for prompt templates"""
    name: str
    template: str

    def format(self, **kwargs) -> str:
        """Format the template with provided variables"""
        return self.template.format(**kwargs)


class MedicalPrescriptionTemplates:
    """Prompt templates for medical prescription domain"""

    # Basic RAG template
    RAG_TEMPLATE = """You are a medical prescription assistant. Your role is to provide accurate, well-cited information about medications, drug interactions, dosages, and prescribing guidelines based on the provided context.

CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the context.
2. If the context doesn't contain sufficient information, state that clearly.
3. Be concise but thorough.
4. Include specific details like dosages, contraindications, or side effects when mentioned in the context.
5. Do not make up information beyond what's in the context.

ANSWER:"""

    # Drug information template
    DRUG_INFO_TEMPLATE = """You are a medical prescription assistant. Provide detailed information about the requested drug based on the context.

DRUG: {entity}
CONTEXT: {context}

Please provide:
1. Indications and usage
2. Dosage and administration
3. Contraindications
4. Adverse reactions/side effects
5. Drug interactions
6. Warnings and precautions

If any information is not available in the context, state that it's not mentioned.

ANSWER:"""

    # Drug interaction template
    DRUG_INTERACTION_TEMPLATE = """You are a medical prescription assistant specializing in drug interactions.

DRUGS TO CHECK: {entity}
CONTEXT: {context}

Identify potential drug interactions from the context. For each interaction found, provide:
1. The drugs involved
2. Severity level (Mild, Moderate, Severe)
3. Description of the interaction
4. Clinical recommendations if mentioned

If no interactions are found in the context, state that clearly.

INTERACTION REPORT:"""

    # Dosage guidance template
    DOSAGE_TEMPLATE = """You are a medical prescription assistant providing dosage guidance.

DRUG: {entity}
CONTEXT: {context}

Based on the context, provide:
1. Recommended dosage for adults
2. Recommended dosage for pediatric patients (if applicable)
3. Dosage adjustments for special populations (elderly, renal/hepatic impairment)
4. Administration instructions
5. Maximum daily dose limits

If information is not available in the context, state that clearly.

DOSAGE GUIDANCE:"""

    # Contraindications template
    CONTRAINDICATIONS_TEMPLATE = """You are a medical prescription assistant.

DRUG: {entity}
CONTEXT: {context}

List all contraindications for this drug based on the context, including:
1. Medical conditions
2. Age restrictions
3. Pregnancy/breastfeeding considerations
4. Concurrent medications
5. Other specific contraindications

CONTRAINDICATIONS:"""

    # Side effects template
    SIDE_EFFECTS_TEMPLATE = """You are a medical prescription assistant.

DRUG: {entity}
CONTEXT: {context}

List side effects for this drug based on the context, categorized by:
1. Common side effects
2. Serious/severe side effects
3. Rare side effects
4. Side effects requiring immediate medical attention

SIDE EFFECTS:"""

    # Prescription summary template
    PRESCRIPTION_SUMMARY_TEMPLATE = """You are a medical prescription assistant. Create a patient-friendly summary.

DRUG: {entity}
CONTEXT: {context}

Create a patient-friendly summary including:
1. What the medication is for
2. How and when to take it
3. Common side effects to watch for
4. Important warnings or precautions
5. When to seek medical help

Use simple, clear language suitable for patients.

PATIENT SUMMARY:"""

    @classmethod
    def get_template(cls, template_type: str) -> PromptTemplate:
        """Get a prompt template by type"""
        templates = {
            "rag": PromptTemplate("RAG", cls.RAG_TEMPLATE),
            "drug_info": PromptTemplate("DrugInfo", cls.DRUG_INFO_TEMPLATE),
            "drug_interaction": PromptTemplate("DrugInteraction", cls.DRUG_INTERACTION_TEMPLATE),
            "dosage": PromptTemplate("Dosage", cls.DOSAGE_TEMPLATE),
            "contraindications": PromptTemplate("Contraindications", cls.CONTRAINDICATIONS_TEMPLATE),
            "side_effects": PromptTemplate("SideEffects", cls.SIDE_EFFECTS_TEMPLATE),
            "prescription_summary": PromptTemplate("PrescriptionSummary", cls.PRESCRIPTION_SUMMARY_TEMPLATE),
        }
        return templates.get(template_type, templates["rag"])

    @classmethod
    def format_rag_prompt(cls, query: str, context: str) -> str:
        """Format a basic RAG prompt"""
        template = cls.get_template("rag")
        return template.format(query=query, context=context)

    @classmethod
    def format_drug_info_prompt(cls, entity: str, context: str) -> str:
        """Format a drug information prompt"""
        template = cls.get_template("drug_info")
        return template.format(entity=entity, context=context)

    @classmethod
    def format_drug_interaction_prompt(cls, entity: str, context: str) -> str:
        """Format a drug interaction prompt"""
        template = cls.get_template("drug_interaction")
        return template.format(entity=entity, context=context)

    @classmethod
    def format_dosage_prompt(cls, entity: str, context: str) -> str:
        """Format a dosage guidance prompt"""
        template = cls.get_template("dosage")
        return template.format(entity=entity, context=context)

    @classmethod
    def format_contraindications_prompt(cls, entity: str, context: str) -> str:
        template = cls.get_template("contraindications")
        return template.format(entity=entity, context=context)

    @classmethod
    def format_side_effects_prompt(cls, entity: str, context: str) -> str:
        template = cls.get_template("side_effects")
        return template.format(entity=entity, context=context)

    @classmethod
    def format_prescription_summary_prompt(cls, entity: str, context: str) -> str:
        template = cls.get_template("prescription_summary")
        return template.format(entity=entity, context=context)
