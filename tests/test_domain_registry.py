from __future__ import annotations

import pytest

from domain_slm_guardrails.core.domain_registry import get_domain_config, list_domains


def test_medical_billing_config_loads():
    domain = get_domain_config("medical_billing")
    assert domain.domain_id == "medical_billing"
    assert domain.name == "Medical Billing"
    assert domain.corpus_path.name == "medical_billing"
    assert domain.index_name == "medical_billing_chunks"


def test_list_domains_includes_medical_billing():
    assert "medical_billing" in list_domains()


def test_unknown_domain_fails_clearly():
    with pytest.raises(ValueError, match="Unknown domain"):
        get_domain_config("missing_domain")

