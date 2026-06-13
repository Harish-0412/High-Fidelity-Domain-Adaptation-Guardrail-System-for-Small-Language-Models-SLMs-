from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProcessedQuery:
    original: str
    cleaned: str
    expanded: str
    intent: str = "factual"


class QueryPreprocessor:
    """
    Query preprocessor for medical prescription domain, with:
    - Text cleaning
    - Medical abbreviation expansion
    - Query expansion (synonyms
    """

    # Common medical abbreviations
    MEDICAL_ABBREVIATIONS = {
        "asprin": "aspirin",
        "paracetamol": "acetaminophen",
        "ibuprophen": "ibuprofen",
        "acetaminophen": "paracetamol",
        "dr": "doctor",
        "dr.": "doctor",
        "md": "medical doctor",
        "po": "by mouth",
        "prn": "as needed",
        "qid": "four times daily",
        "tid": "three times daily",
        "bid": "twice daily",
        "od": "once daily",
        "qd": "once daily",
        "hs": "at bedtime",
        "ac": "before meals",
        "pc": "after meals",
    }

    # Common stop words (lightweight)
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "i", "me", "my", "you", "your", "he", "she", "it",
        "what", "when", "where", "why", "how",
    }

    def __init__(self, domain: Optional[str] = "medical_prescription"):
        self.domain = domain

    def preprocess(
        self,
        query: str,
    ) -> ProcessedQuery:
        cleaned = self._clean_query(query)
        expanded = self._expand_query(cleaned)
        return ProcessedQuery(
            original=query,
            cleaned=cleaned,
            expanded=expanded,
        )

    def _clean_query(self, query: str) -> str:
        cleaned = query.strip().lower()
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        return cleaned

    def _expand_query(self, cleaned: str) -> str:
        words = cleaned.split()
        expanded_words = []
        for word in words:
            if word in self.STOP_WORDS:
                expanded_words.append(word)
                continue
            if word in self.MEDICAL_ABBREVIATIONS:
                expanded_words.append(word)
                expanded_words.append(self.MEDICAL_ABBREVIATIONS[word])
            else:
                expanded_words.append(word)
        return ' '.join(expanded_words)
