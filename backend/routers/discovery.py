"""
Discovery & Comparison routes.
Feature 1: Natural language → hard filter → deterministic weighted ranking
Feature 2: Multi-policy comparison table
Feature 3 (Chat): 3-mode conversational advisor
  Mode GATHER   — asks smart follow-up questions when any essential field missing
  Mode RECOMMEND — hard filter + weighted rank + RAG insights from actual PDF
  Mode EXPLAIN   — explains insurance terms grounded in actual policy document text
"""
from fastapi import APIRouter
from pydantic import BaseModel
from services import llm, vector_store
from services.skills import PolicyRanker, hard_filter
from services.advisor_agent import (
    classify_intent,
    find_uploaded_for_insurer,
    get_rag_insights,
    explain_term,
    get_chat_reply,
)

router = APIRouter(prefix="/api", tags=["discovery"])
ranker = PolicyRanker()

EXTRACT_REQUIREMENTS_SYSTEM = """Extract health insurance requirements from user query.
Return ONLY valid JSON:
{
  "needs": ["maternity", "diabetes_management", "opd"],
  "budget_max": 18000,
  "members": 3,
  "preexisting_conditions": ["type_2_diabetes"],
  "preferred_type": "family_floater",
  "sum_insured_min": 500000
}
If a field is not mentioned, omit it or use null. needs can include: maternity, opd, mental_health, ayush, dental, critical_illness, restoration, ncb."""

COMPARISON_SYSTEM = """You are a health insurance advisor. Given structured data for 2-3 policies, generate a plain English comparison summary.
Focus on key differences in coverage, waiting periods, and value. Keep it under 150 words. Be specific with numbers.
Return JSON: {"summary": "your comparison text", "best_for": {"policy_name": "reason"}}"""

CHAT_INTRO_SYSTEM = """You are a warm health insurance advisor. Write a friendly 1-2 sentence response acknowledging what the user asked for, right before showing their policy recommendations. Be specific about what you understood. Do not say "Great!" or "Sure!" — be natural.
Return ONLY valid JSON: {"message": "your response here"}"""

NO_RESULTS_MESSAGE = (
    "No policies in our catalog match all your hard requirements exactly. "
    "Try: relaxing your budget, removing a specific coverage requirement, "
    "or changing the plan type. I can help you find the closest match if you adjust any one criterion."
)


class DiscoverRequest(BaseModel):
    query: str


class CompareRequest(BaseModel):
    policy_ids: list[str]


class DiscoverChatRequest(BaseModel):
    messages: list[dict]  # [{role: "user"|"assistant", content: str}]
    session_policy_ids: list[str] = []  # uploaded PDF IDs from last recommendation (for term lookup)


def _apply_hard_filter_and_rank(requirements: dict) -> list[dict]:
    """
    Applies hard_filter first. If 0 survive, returns [].
    No silent fallback — caller decides how to handle empty.
    """
    requirements["needs"] = requirements.get("needs") or []
    requirements["preexisting_conditions"] = requirements.get("preexisting_conditions") or []

    all_policies = vector_store.list_catalog_policies()
    filtered = hard_filter(all_policies, requirements)

    if not filtered:
        return []

    return ranker.rank(requirements, filtered)


@router.post("/discover")
async def discover_policies(req: DiscoverRequest):
    """Extract requirements from natural language, apply hard filter, return deterministic ranked list."""
    requirements = llm.chat_json(EXTRACT_REQUIREMENTS_SYSTEM, req.query)
    ranked = _apply_hard_filter_and_rank(requirements)

    if not ranked:
        return {
            "extracted_requirements": requirements,
            "policies": [],
            "total_found": 0,
            "message": NO_RESULTS_MESSAGE,
        }

    return {
        "extracted_requirements": requirements,
        "policies": ranked[:6],
        "total_found": len(ranked),
    }


@router.post("/discover/chat")
async def discover_chat(req: DiscoverChatRequest):
    """
    3-mode conversational advisor:
      Mode GATHER   — asks smart questions when any essential field (budget/members/needs) is missing
      Mode EXPLAIN  — explains insurance terms grounded in actual uploaded policy PDFs
      Mode RECOMMEND — hard filter + weighted rank + RAG insights per policy from actual PDF text
    """
    if not req.messages:
        return {
            "type": "question",
            "message": "What health coverage are you looking for? Tell me your needs, budget, and family size.",
        }

    # Step 1: Classify intent + extract all requirements from full conversation
    conversation = "\n".join([
        f"{m['role'].upper()}: {m['content']}" for m in req.messages
    ])
    intent_result = classify_intent(conversation)

    intent = intent_result.get("intent", "gather_info")
    extracted = intent_result.get("extracted") or {}
    extracted["needs"] = extracted.get("needs") or []
    extracted["preexisting_conditions"] = extracted.get("preexisting_conditions") or []

    # ── MODE GATHER: any essential field missing ──────────────────────────────
    missing_any = (
        not intent_result.get("has_budget")
        or not intent_result.get("has_members")
        or not intent_result.get("has_needs_or_conditions")
    )
    if intent == "gather_info" or (missing_any and intent not in ("explain_term", "explain_policy", "chat_reply")):
        question = (
            intent_result.get("next_question")
            or "Could you tell me your health coverage needs, annual budget, and how many family members need coverage?"
        )
        return {"type": "question", "message": question}

    # ── MODE CHAT: conversational / educational questions ─────────────────────
    if intent == "chat_reply":
        last_user = next(
            (m["content"] for m in reversed(req.messages) if m["role"] == "user"), ""
        )
        reply = get_chat_reply(last_user, req.session_policy_ids)
        return {"type": "chat", "message": reply["answer"]}

    # ── MODE EXPLAIN: user asked about a term or specific policy ─────────────
    if intent in ("explain_term", "explain_policy"):
        term = intent_result.get("term_to_explain") or intent_result.get("policy_name_asked")
        if term:
            result = explain_term(term, req.session_policy_ids)
            return {
                "type": "explanation",
                "message": result.get("explanation", ""),
                "example": result.get("example"),
                "citation": result.get("citation"),
                "policy_name": result.get("policy_name"),
                "found": result.get("found", False),
            }

    # ── MODE RECOMMEND: all 3 essential fields present ────────────────────────
    ranked = _apply_hard_filter_and_rank(extracted)

    if not ranked:
        return {
            "type": "no_results",
            "message": NO_RESULTS_MESSAGE,
            "extracted_requirements": extracted,
            "policies": [],
            "total_found": 0,
        }

    top_policies = ranked[:6]

    # RAG enrichment: for top 3 policies, find matching uploaded PDF → surface hidden traps
    user_needs = extracted["needs"] + extracted["preexisting_conditions"]
    uploaded_ids: list[str] = []

    for i, policy in enumerate(top_policies[:3]):
        uploaded = find_uploaded_for_insurer(policy.get("insurer", ""))
        if uploaded:
            insights = get_rag_insights(uploaded["id"], user_needs, policy.get("insurer", ""))
            policy["rag_insights"] = insights
            policy["uploaded_policy_id"] = uploaded["id"]
            uploaded_ids.append(uploaded["id"])
        else:
            policy["rag_insights"] = {"available": False}

    # Build contextual intro message
    last_user = next(
        (m["content"] for m in reversed(req.messages) if m["role"] == "user"), ""
    )
    intro_result = llm.chat_json(
        CHAT_INTRO_SYSTEM,
        f"User asked: {last_user}\nExtracted needs: {extracted}",
    )
    message = intro_result.get("message") or "Here are the best policies matching your needs:"

    return {
        "type": "results",
        "message": message,
        "extracted_requirements": extracted,
        "policies": top_policies,
        "total_found": len(ranked),
        "uploaded_policy_ids": uploaded_ids,  # client stores these for future term lookups
    }


@router.post("/compare")
async def compare_policies(req: CompareRequest):
    """Return side-by-side comparison of 2-3 policies."""
    if len(req.policy_ids) < 2:
        return {"error": "Please provide at least 2 policy IDs to compare."}
    if len(req.policy_ids) > 3:
        req.policy_ids = req.policy_ids[:3]

    policies = [
        p for pid in req.policy_ids
        if (p := vector_store.get_catalog_policy(pid)) is not None
    ]

    if len(policies) < 2:
        return {"error": "Could not find the requested policies."}

    fields = [
        ("insurer", "Insurer"),
        ("type", "Plan Type"),
        ("premium_min", "Min Premium (₹/yr)"),
        ("premium_max", "Max Premium (₹/yr)"),
        ("sum_insured_min", "Min Sum Insured (₹)"),
        ("sum_insured_max", "Max Sum Insured (₹)"),
        ("waiting_period_preexisting_years", "Pre-existing Wait (years)"),
        ("waiting_period_maternity_months", "Maternity Wait (months)"),
        ("co_pay_percent", "Co-pay (%)"),
        ("room_rent_limit", "Room Rent Limit"),
        ("covers_maternity", "Maternity Coverage"),
        ("covers_opd", "OPD Coverage"),
        ("covers_mental_health", "Mental Health"),
        ("covers_ayush", "AYUSH Coverage"),
        ("covers_dental", "Dental Coverage"),
        ("daycare_procedures", "Daycare Procedures"),
        ("ncb_percent", "No Claim Bonus (%)"),
        ("restoration_benefit", "Restoration Benefit"),
        ("network_hospitals", "Network Hospitals"),
    ]

    comparison_rows = []
    for field_key, field_label in fields:
        row = {"dimension": field_label}
        for p in policies:
            val = p.get(field_key)
            if isinstance(val, bool):
                val = "Yes" if val else "No"
            elif val is None:
                val = "—"
            row[p["name"]] = val
        comparison_rows.append(row)

    policy_summary = "\n\n".join([
        f"Policy: {p['name']} ({p['insurer']})\n"
        f"Premium: ₹{p.get('premium_min', 0):,}–{p.get('premium_max', 0):,}/yr | "
        f"PED wait: {p.get('waiting_period_preexisting_years', '?')} yrs | "
        f"Maternity: {'Yes' if p.get('covers_maternity') else 'No'} | "
        f"OPD: {'Yes' if p.get('covers_opd') else 'No'} | "
        f"Network: {p.get('network_hospitals', 0):,} hospitals | "
        f"Restoration: {'Yes' if p.get('restoration_benefit') else 'No'}"
        for p in policies
    ])

    ai_summary = llm.chat_json(COMPARISON_SYSTEM, f"Compare these policies:\n{policy_summary}")

    return {
        "policies": [{"id": p["id"], "name": p["name"], "insurer": p["insurer"]} for p in policies],
        "comparison_table": comparison_rows,
        "ai_summary": ai_summary.get("summary", ""),
        "best_for": ai_summary.get("best_for", {}),
    }
