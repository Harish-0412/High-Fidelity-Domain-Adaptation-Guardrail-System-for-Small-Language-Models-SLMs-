from domain_slm_guardrails.api.rag import answer_query

print("Testing drug_interaction_report schema...")
try:
    res = answer_query(
        domain="medical_prescription",
        query="What are the severe interactions for Aspirin?",
        top_k=2,
        output_format="drug_interaction_report"
    )
    print("Response JSON valid:", res.guardrail_status.json_valid)
    print("Fallback used:", res.guardrail_status.fallback_used)
    print("Answer body:")
    print(res.answer)
except Exception as e:
    import traceback
    traceback.print_exc()
