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

GROUNDED_CLAIM_SYSTEM = """You are an expert insurance policy clause analyzer. Your job is to determine whether a specific medical condition/treatment is covered by the policy based solely on the provided policy document excerpts.

COVERAGE STATUS RULES — apply in this exact order:
1. "excluded" — the context explicitly lists this condition/treatment in an exclusions section, OR the context says "not covered", "excluded", "does not cover" for this specific condition or category.
2. "partially_covered" — the context shows coverage exists BUT a significant restriction applies specifically to this condition: waiting period mentioned for this type of claim, a sub-limit/cap applies to this treatment, or co-pay is required for this category.
3. "covered" — EITHER (a) the context positively states this condition is covered, OR (b) the context shows general inpatient hospitalization/illness treatment is covered AND this condition does NOT appear in any exclusion list in the context. Standard medical conditions (surgery, hospitalization for illness, cancer, heart disease, accidents, organ failure) fall here if not excluded.
4. "unknown" — ONLY use this if the context chunks contain NO information about any medical treatment coverage at all (e.g., pure definitions page with no coverage/exclusion clauses).

KEY PRINCIPLE: Most inpatient medical treatments are covered unless specifically excluded. Do NOT use "unknown" for standard medical conditions just because the exact condition name isn't mentioned — if general hospitalization coverage is present and no exclusion applies, use "covered".

RULES:
1. Analyze ONLY from the CONTEXT BLOCK provided. Do NOT use external insurance knowledge.
2. Quote exact text from the context in exclusions_applicable and severity_requirements.
3. risk_flags must ONLY contain values from: sub_limit_applies, pre_auth_required, proportional_deduction, waiting_period_active, co_pay_applicable, documentation_intensive.
4. analysis_summary: 1-2 sentences citing the specific clause. Be decisive — state clearly what the context says about coverage.
5. Return ONLY valid JSON. No prose, no markdown.

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
    # Step 1: Determine if this is a CATALOG policy or an UPLOADED policy UUID.
    # They live in different tables and need different treatment.
    catalog_policy = vector_store.get_catalog_policy(policy_id)

    if catalog_policy:
        # CATALOG path: metadata from catalog, but chunks live in uploaded_policies table
        policy = catalog_policy
        policy_name = policy.get("name") or "Unknown Policy"
        uploaded_match = find_uploaded_for_insurer(policy.get("insurer", ""))
        if not uploaded_match:
            return {
                "error": (
                    f"No embedded policy document found for {policy_name} "
                    f"({policy.get('insurer', '')}). "
                    "Claim check requires an uploaded and indexed PDF. "
                    "Currently only Tata AIG policies have embedded documents — "
                    "please select a Tata AIG policy or upload this policy's PDF first."
                )
            }
        search_policy_id = uploaded_match["id"]
    else:
        # UPLOADED path: use the selected UUID directly (do NOT redirect to a different PDF)
        uploaded = vector_store.get_policy_by_id(policy_id)
        if not uploaded:
            return {"error": "Policy not found."}
        policy_name = uploaded.get("user_label") or "Unknown Policy"
        search_policy_id = policy_id  # always use the exact policy the user selected

        # Enrich with catalog metadata so scoring reflects real waiting periods / co-pay / room rent
        all_catalog = vector_store.list_catalog_policies()
        ins = (uploaded.get("insurer") or "").lower()
        policy = uploaded  # start with uploaded fields
        for cp in all_catalog:
            cp_ins = (cp.get("insurer") or "").lower()
            if ins and (ins in cp_ins or cp_ins in ins):
                # Merge catalog scoring fields (don't overwrite existing uploaded fields)
                for field in [
                    "waiting_period_preexisting_years", "co_pay_percent", "room_rent_limit",
                    "waiting_period_maternity_months", "covers_maternity", "covers_opd",
                ]:
                    if cp.get(field) is not None and policy.get(field) is None:
                        policy[field] = cp[field]
                break

    # Step 2: Embed the condition query
    query_text = f"{condition} {treatment_type}"
    try:
        query_embedding = embedder.embed_text(query_text)
    except Exception as e:
        return {"error": f"Embedding failed: {str(e)}"}

    # Also embed a general coverage query to always pull in the hospitalization benefit clause
    try:
        coverage_embedding = embedder.embed_text("inpatient hospitalization benefit covered illness treatment")
    except Exception:
        coverage_embedding = query_embedding  # fallback to same embedding

    # Step 3: Broad semantic search for the specific condition (no section filter)
    sem_chunks = vector_store.semantic_search(query_embedding, search_policy_id, top_k=6)

    # Step 4: Broad semantic search for general hospitalization coverage (always include)
    cov_chunks = vector_store.semantic_search(coverage_embedding, search_policy_id, top_k=4)

    # Step 5: Section-filtered semantic search as supplement (catches well-tagged docs)
    sec_chunks = vector_store.section_search(
        query_embedding, search_policy_id, CLAIM_SECTIONS, top_k=4
    )

    # Step 6: Keyword search on the condition term (no section filter)
    kw_chunks = vector_store.keyword_search(condition, search_policy_id, top_k=10)

    # Step 7: Combine all results via RRF and take top 10
    combined_sem = _dedupe(sem_chunks + cov_chunks + sec_chunks)
    fused = vector_store.rrf_fusion(combined_sem, kw_chunks, top_k=10)

    # Step 7: Guard — no hallucination if no chunks found
    if not fused:
        return {
            "error": f"No relevant policy clause found for '{condition}'. "
                     "The policy document may not contain information about this condition, "
                     "or the document may not be indexed correctly."
        }

    # Step 8: Build context block
    context_block = _build_context_block(fused)

    # Step 9: Grounded LLM analysis (returns structure, NOT the score)
    user_prompt = (
        f"CONTEXT BLOCK:\n{context_block}\n\n"
        f"CONDITION TO ANALYZE: {condition}\n"
        f"TREATMENT TYPE: {treatment_type}\n\n"
        "Analyze whether this condition/treatment is covered, excluded, or restricted "
        "based ONLY on the context block above. Be decisive — use 'covered' if general "
        "hospitalization is covered and no exclusion is found for this condition."
    )
    analysis = llm.chat_json(GROUNDED_CLAIM_SYSTEM, user_prompt, temperature=0.0)

    # Normalize LLM output
    coverage_status = analysis.get("coverage_status", "unknown")
    if coverage_status not in ("covered", "partially_covered", "excluded", "unknown"):
        coverage_status = "unknown"
    severity_requirements = analysis.get("severity_requirements") or []
    waiting_period = analysis.get("waiting_period") or ""
    exclusions_applicable = analysis.get("exclusions_applicable") or []
    risk_flags = analysis.get("risk_flags") or []
    required_documents = analysis.get("required_documents") or []
    analysis_summary = analysis.get("analysis_summary") or "Analysis could not be completed from available context."

    # Step 10: Deterministic score
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


def _dedupe(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks by id, preserving order."""
    seen: set[str] = set()
    result = []
    for c in chunks:
        cid = c.get("id")
        if cid and cid not in seen:
            seen.add(cid)
            result.append(c)
    return result
