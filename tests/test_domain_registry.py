from __future__ import annotations

import pytest

from domain_slm_guardrails.core.domain_registry import get_domain_config, list_domains


def test_medical_prescription_config_loads():
    domain = get_domain_config("medical_prescription")
    assert domain.domain_id == "medical_prescription"
    assert domain.name == "Medical Prescription"
    assert domain.corpus_path.name == "medical_prescription"
    assert domain.index_name == "medical_prescription_chunks"


def test_list_domains_includes_medical_prescription():
    assert "medical_prescription" in list_domains()


def test_unknown_domain_fails_clearly():
    with pytest.raises(ValueError, match="Unknown domain"):
        get_domain_config("missing_domain")
