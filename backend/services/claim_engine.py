"""
Deterministic, RAG-grounded claim check engine.

Pipeline:
  1. Section-filtered semantic search (exclusions, coverage, waiting_periods, conditions, limits)
  2. Full keyword search → post-filter by same sections
  3. RRF fusion → top 8 chunks
  4. If 0 chunks: return structured error (no hallucination)
  5. Build CONTEXT BLOCK → strict grounded LLM analysis
  6. Deterministic compute_claim_score() — LLM does NOT set the score
  7. Return structured result
"""
from services import llm, vector_store, embedder
from services.advisor_agent import find_uploaded_for_insurer

CLAIM_SECTIONS = ["exclusions", "coverage", "waiting_periods", "conditions", "limits"]

GROUNDED_CLAIM_SYSTEM = """You are a strict insurance policy clause analyzer.

RULES — READ CAREFULLY:
1. Analyze ONLY from the CONTEXT BLOCK provided. Do NOT use external insurance knowledge.
2. If the context does not mention the condition or treatment, set coverage_status to "unknown".
3. Do NOT fabricate clauses, waiting periods, or exclusions that are not in the context.
4. Quote exact text from the context in exclusions_applicable and severity_requirements.
5. risk_flags must ONLY contain values from: sub_limit_applies, pre_auth_required, proportional_deduction, waiting_period_active, co_pay_applicable, documentation_intensive.
6. analysis_summary must cite the specific clause or page reference from context.
7. Return ONLY valid JSON. No prose, no markdown, no explanation outside the JSON.

Return this exact JSON schema:
{
  "coverage_status": "covered | partially_covered | excluded | unknown",
  "severity_requirements": ["exact quoted criteria from context"],
  "waiting_period": "exact waiting period text from context, or empty string if none found",
  "exclusions_applicable": ["exact exclusion clause text from context"],
  "risk_flags": ["sub_limit_applies", "pre_auth_required", ...],
  "required_documents": ["discharge summary", "..."],
  "analysis_summary": "1-2 sentences citing specific clause or section"
}"""


def _build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk.get("section_type", "general").upper()
        page = chunk.get("page_number", "?")
        parts.append(f"[CHUNK {i} | Section: {section} | Page: {page}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def compute_claim_score(
    coverage_status: str,
    exclusions_applicable: list,
    risk_flags: list,
    policy: dict,
) -> int:
    """
    Fully deterministic scoring — LLM output informs inputs but does NOT set the score.

    Scoring breakdown:
      Coverage base:
        +50  covered
        +25  partially_covered
        +0   excluded / unknown

      Waiting period (from policy metadata):
        +20  PED wait <= 1 year
        +12  PED wait <= 2 years
        +0   PED wait 3 years
        -0   (no penalty for 3; penalty applied via risk flag)

      Exclusion status:
        +15  no applicable exclusions found
        -25  explicit exclusion clause found

      Risk flags:
        +10  no risk flags at all
        -5   per risk flag (capped at -20)

      Policy metadata penalties:
        -5   co-pay > 0%
        -10  room rent limit contains "%" (proportional deduction risk)
        -5   room rent limit is a fixed cap (not "No limit")
    """
    score = 0

    # Coverage base
    status = coverage_status.lower()
    if status == "covered":
        score += 50
    elif status == "partially_covered":
        score += 25
    # excluded / unknown: 0

    # Waiting period from metadata
    ped = policy.get("waiting_period_preexisting_years", 4)
    if ped <= 1:
        score += 20
    elif ped <= 2:
        score += 12

    # Exclusions
    excl = exclusions_applicable or []
    if not excl:
        score += 15
    else:
        score -= 25

    # Risk flags
    flags = risk_flags or []
    if not flags:
        score += 10
    else:
        score -= min(len(flags) * 5, 20)

    # Policy metadata penalties
    if policy.get("co_pay_percent", 0) > 0:
        score -= 5
    room = policy.get("room_rent_limit") or ""
    if room and "%" in room:
        score -= 10
    elif room and room.lower() not in ("no limit", "no sub-limits", "no restriction", ""):
        score -= 5

    return max(0, min(100, score))


def _get_policy_metadata(policy_id: str) -> dict:
    """Try catalog first, then uploaded policies. Returns empty dict if not found."""
    policy = vector_store.get_catalog_policy(policy_id)
    if policy:
        return policy
    uploaded = vector_store.get_policy_by_id(policy_id)
    return uploaded or {}


def run_claim_check(policy_id: str, condition: str, treatment_type: str) -> dict:
    """
    Full claim check pipeline.

    Returns either:
      {"error": "..."} — if no relevant chunks found
    or:
      {structured result dict}
    """
    # Step 1: Get policy metadata for scoring + display
    policy = _get_policy_metadata(policy_id)
    policy_name = (
        policy.get("name")
        or policy.get("user_label")
        or "Unknown Policy"
    )

    # Step 1b: Resolve which UUID to use for chunk search.
    # Catalog policies (insurance_policies table) have an "insurer" field but NO chunks in
    # policy_chunks. Chunks are stored under uploaded_policies UUIDs. Map via insurer name.
    search_policy_id = policy_id  # default: assume it's already an uploaded policy UUID
    if policy.get("insurer"):
        # This is a catalog policy — find the matching uploaded PDF
        uploaded_match = find_uploaded_for_insurer(policy["insurer"])
        if uploaded_match:
            search_policy_id = uploaded_match["id"]
        else:
            return {
                "error": (
                    f"No embedded policy document found for {policy.get('name', 'this policy')} "
                    f"({policy.get('insurer', '')}). "
                    "Claim check requires an uploaded and indexed PDF. "
                    "Currently only Tata AIG policies have embedded documents — "
                    "please select a Tata AIG policy or upload this policy's PDF first."
                )
            }

    # Step 2: Embed the condition query
    query_text = f"{condition} {treatment_type} coverage exclusion waiting period"
    try:
        query_embedding = embedder.embed_text(query_text)
    except Exception as e:
        return {"error": f"Embedding failed: {str(e)}"}

    # Step 3: Section-filtered semantic search (against uploaded PDF chunks)
    sem_chunks = vector_store.section_search(
        query_embedding, search_policy_id, CLAIM_SECTIONS, top_k=6
    )

    # Step 4: Keyword search → post-filter to relevant sections
    kw_all = vector_store.keyword_search(condition, search_policy_id, top_k=10)
    kw_chunks = [c for c in kw_all if c.get("section_type") in CLAIM_SECTIONS]

    # Step 5: RRF fusion → top 8
    fused = vector_store.rrf_fusion(sem_chunks, kw_chunks, top_k=8)

    # Step 6: Guard — no hallucination if no chunks found
    if not fused:
        return {
            "error": f"No relevant policy clause found for '{condition}'. "
                     "The policy document may not contain information about this condition, "
                     "or the document may not be indexed correctly."
        }

    # Step 7: Build context block
    context_block = _build_context_block(fused)

    # Step 8: Grounded LLM analysis (returns structure, NOT the score)
    user_prompt = (
        f"CONTEXT BLOCK:\n{context_block}\n\n"
        f"CONDITION TO ANALYZE: {condition}\n"
        f"TREATMENT TYPE: {treatment_type}\n\n"
        "Analyze coverage for the above condition using ONLY the context block above."
    )
    analysis = llm.chat_json(GROUNDED_CLAIM_SYSTEM, user_prompt, temperature=0.0)

    # Normalize LLM output (guard against None/missing fields)
    coverage_status = analysis.get("coverage_status", "unknown")
    if coverage_status not in ("covered", "partially_covered", "excluded", "unknown"):
        coverage_status = "unknown"
    severity_requirements = analysis.get("severity_requirements") or []
    waiting_period = analysis.get("waiting_period") or ""
    exclusions_applicable = analysis.get("exclusions_applicable") or []
    risk_flags = analysis.get("risk_flags") or []
    required_documents = analysis.get("required_documents") or []
    analysis_summary = analysis.get("analysis_summary") or "Analysis could not be completed from available context."

    # Step 9: Deterministic score (LLM output feeds inputs, not the score itself)
    feasibility_score = compute_claim_score(
        coverage_status, exclusions_applicable, risk_flags, policy
    )

    return {
        "policy_name": policy_name,
        "diagnosis": condition,
        "treatment_type": treatment_type,
        "coverage_status": coverage_status,
        "feasibility_score": feasibility_score,
        "severity_requirements": severity_requirements,
        "waiting_period": waiting_period,
        "exclusions_applicable": exclusions_applicable,
        "risk_flags": risk_flags,
        "required_documents": required_documents,
        "analysis_summary": analysis_summary,
        "chunks_used": len(fused),
        "error": None,
    }
