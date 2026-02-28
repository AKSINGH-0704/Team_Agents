"""
Intelligent Discovery Advisor Agent — 3 core capabilities:

1. classify_intent()       — LLM classifies user intent + checks all 3 essential fields
2. find_uploaded_for_insurer() — fuzzy match catalog insurer → uploaded PDF
3. get_rag_insights()      — section-filtered RAG → hidden traps from actual PDF text
4. explain_term()          — RAG lookup of insurance term in definitions/conditions sections
"""
from services import llm, vector_store, embedder

# ─── Prompts ─────────────────────────────────────────────────────────────────

ADVISOR_INTENT_SYSTEM = """You are classifying a user's health insurance query intent and extracting key information.

Essential fields for a good policy recommendation (ALL 3 required):
  1. budget_max — annual premium in INR (any mention of money/year, e.g. "10k/yr", "₹15,000 budget")
  2. members — number of people needing coverage (self, family size, "just me")
  3. needs_or_conditions — at least ONE health need or condition (maternity, diabetes, OPD, mental health, etc.)

Return ONLY valid JSON — no prose, no markdown:
{
  "intent": "gather_info | recommend_policies | explain_term | explain_policy | refine_results | chat_reply",
  "has_budget": true or false,
  "has_members": true or false,
  "has_needs_or_conditions": true or false,
  "next_question": "one specific question for the HIGHEST-PRIORITY missing field, or null if all present",
  "term_to_explain": "exact insurance term user asked about (e.g. room rent limit, co-pay, NCB, waiting period), or null",
  "policy_name_asked": "policy name user asked about, or null",
  "extracted": {
    "needs": [],
    "budget_max": null,
    "members": null,
    "preexisting_conditions": [],
    "preferred_type": null,
    "sum_insured_min": null
  }
}

Intent classification rules:
- "chat_reply": user is greeting, making small talk, asking a general educational insurance question, asking for advice without wanting specific recommendations, or saying something conversational (hi, thanks, what is health insurance, tell me about waiting periods, which plan is best for elderly, etc.) — answer naturally in text
- "gather_info": user wants policy recommendations but ANY of the 3 essential fields is missing → ask next_question
- "recommend_policies": ALL 3 fields present AND user clearly wants policy recommendations
- "explain_term": user asks to explain/define a specific insurance term (room rent, co-pay, NCB, waiting period, cashless, sub-limit, TPA, daycare, etc.)
- "explain_policy": user asks for more detail about a specific named policy
- "refine_results": user wants to change/narrow results already shown

chat_reply examples: "hi", "hello", "thanks", "what is health insurance?", "which type of plan is better?", "tell me about waiting periods", "is maternity coverage worth it?", "what should I look for in a policy?", "are there any good plans for senior citizens?"

next_question priority (ask ONLY ONE at a time):
  1. If no needs_or_conditions → "What specific health coverage do you need? For example: maternity, diabetes management, OPD visits, or a critical illness plan?"
  2. If no budget → "What's your annual premium budget? For example ₹10,000/year or ₹20,000/year."
  3. If no members → "How many family members need to be covered, including yourself?"

Budget extraction: treat "10k", "15,000", "₹20k/yr", "under 12000 per year" all as budget_max numbers.
Members extraction: "just me" = 1, "me and my wife" = 2, "family of 4" = 4, "self and 2 kids" = 3.
Needs extraction: map to canonical values: maternity, opd, mental_health, ayush, dental, critical_illness, restoration, ncb.
Sum insured extraction: "₹10L coverage", "10 lakh cover", "minimum 5L" → sum_insured_min as integer in INR."""


RAG_INSIGHTS_SYSTEM = """You are analyzing health insurance policy clauses to find hidden conditions and key facts.

RULES — STRICTLY FOLLOW:
1. Analyze ONLY from the CONTEXT BLOCK provided below.
2. Do NOT use general insurance knowledge.
3. Report ONLY conditions explicitly mentioned in the provided text.
4. Keep all explanations in plain, simple English (no jargon).
5. If context does not have enough information, return grounded=false.

Return ONLY valid JSON:
{
  "hidden_traps": [
    {
      "type": "room_rent_trap | proportional_deduction | pre_auth_required | sub_limit | waiting_period | definition_trap",
      "plain_english": "one sentence any person can understand",
      "impact": "what this means for your claim — in rupee terms if possible"
    }
  ],
  "key_fact": "the single most important thing to know about this policy from the clauses, or null",
  "grounded": true
}

If context lacks relevant content: {"hidden_traps": [], "key_fact": null, "grounded": false}"""


EXPLAIN_TERM_SYSTEM = """You are explaining an insurance term to someone with no insurance knowledge.

RULES — STRICTLY FOLLOW:
1. Use ONLY the CONTEXT BLOCK provided below.
2. Write in 2-3 simple sentences that a 10-year-old could understand.
3. Give a concrete example using rupee amounts (₹).
4. Quote the exact relevant clause from the context.
5. If the context does not define or describe this term, set found=false.

Return ONLY valid JSON:
{
  "found": true or false,
  "explanation": "plain English explanation in 2-3 sentences",
  "example": "concrete example with rupee amounts, e.g. If your hospital room costs ₹5,000/day but limit is 1% of ₹5L SI = ₹5,000/day, so you're fine. But if room costs ₹8,000/day, you pay the ₹3,000 difference AND proportional cuts to other charges.",
  "citation": "exact quoted clause from context",
  "policy_name": null
}"""


# ─── Context Block Builder ────────────────────────────────────────────────────

def _build_context_block(chunks: list[dict]) -> str:
    """Build structured CONTEXT BLOCK string from chunks (same format as claim_engine)."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk.get("section_type", "general").upper()
        page = chunk.get("page_number", "?")
        parts.append(f"[CHUNK {i} | Section: {section} | Page: {page}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


# ─── Intent Classifier ────────────────────────────────────────────────────────

def classify_intent(conversation: str) -> dict:
    """
    Classify user intent and extract requirements from full conversation text.

    Returns dict with keys:
      intent, has_budget, has_members, has_needs_or_conditions,
      next_question, term_to_explain, policy_name_asked, extracted
    """
    result = llm.chat_json(ADVISOR_INTENT_SYSTEM, f"Conversation:\n{conversation}")
    # Normalize extracted sub-dict
    extracted = result.get("extracted") or {}
    extracted["needs"] = extracted.get("needs") or []
    extracted["preexisting_conditions"] = extracted.get("preexisting_conditions") or []
    result["extracted"] = extracted
    return result


# ─── Insurer → Uploaded PDF Matcher ──────────────────────────────────────────

def find_uploaded_for_insurer(insurer_name: str) -> dict | None:
    """
    Find an uploaded policy whose insurer fuzzy-matches the catalog insurer name.
    Uses first meaningful word of the insurer name (e.g. "Tata" from "Tata AIG General Insurance").

    Returns uploaded policy dict {id, user_label, insurer, chunk_count} or None.
    """
    if not insurer_name:
        return None

    client = vector_store.get_client()
    words = [w for w in insurer_name.split() if len(w) > 2]
    if not words:
        return None

    # First word usually identifies the insurer uniquely (Tata, HDFC, ICICI, Star, Niva, etc.)
    search_term = words[0]

    try:
        result = (
            client.table("uploaded_policies")
            .select("id, user_label, insurer, chunk_count")
            .ilike("insurer", f"%{search_term}%")
            .gt("chunk_count", 0)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception:
        return None


# ─── RAG Insights ─────────────────────────────────────────────────────────────

INSIGHT_SECTIONS = ["exclusions", "conditions", "limits", "waiting_periods", "coverage"]


def get_rag_insights(uploaded_policy_id: str, user_needs: list[str], insurer: str = "") -> dict:
    """
    Run section-filtered RAG on an uploaded policy PDF to surface hidden conditions
    relevant to the user's stated needs.

    Returns:
      {"available": True, "hidden_traps": [...], "key_fact": "...", "grounded": True, "policy_id": "..."}
      or {"available": False} on error/no chunks.
    """
    if not user_needs:
        user_needs = ["coverage", "hospitalization"]

    query = f"{' '.join(user_needs)} coverage exclusion waiting period room rent co-pay sub-limit"

    # Embed query
    try:
        query_emb = embedder.embed_text(query)
    except Exception:
        return {"available": False}

    # Section-filtered semantic search
    sem = vector_store.section_search(
        query_emb, uploaded_policy_id, INSIGHT_SECTIONS, top_k=5
    )

    # Keyword search → filter to relevant sections
    kw_all = vector_store.keyword_search(query, uploaded_policy_id, top_k=8)
    kw = [c for c in kw_all if c.get("section_type") in INSIGHT_SECTIONS]

    # RRF fusion
    fused = vector_store.rrf_fusion(sem, kw, top_k=6)

    if not fused:
        return {"available": False}

    context = _build_context_block(fused)
    result = llm.chat_json(
        RAG_INSIGHTS_SYSTEM,
        f"CONTEXT BLOCK:\n{context}\n\nUSER NEEDS: {user_needs}",
        temperature=0.0,
    )

    # Normalize
    result["available"] = True
    result["policy_id"] = uploaded_policy_id
    result.setdefault("hidden_traps", [])
    result.setdefault("key_fact", None)
    result.setdefault("grounded", False)
    return result


# ─── Term Explanation via RAG ─────────────────────────────────────────────────

EXPLAIN_SECTIONS = ["definitions", "conditions", "limits"]


def explain_term(term: str, session_policy_ids: list[str]) -> dict:
    """
    Look up an insurance term in definitions/conditions sections of the session's
    recommended policies (top 3). Returns first grounded explanation found.

    Returns dict with: found, explanation, example, citation, policy_name
    """
    if not term or not session_policy_ids:
        return _not_found(term)

    for policy_id in session_policy_ids[:3]:
        # Embed the term
        try:
            query_emb = embedder.embed_text(term)
        except Exception:
            continue

        # Section-filtered semantic search for definitions
        def_chunks = vector_store.section_search(
            query_emb, policy_id, EXPLAIN_SECTIONS, top_k=4
        )

        # Keyword search → filter to definition/condition sections
        kw_all = vector_store.keyword_search(term, policy_id, top_k=6)
        kw_filtered = [c for c in kw_all if c.get("section_type") in EXPLAIN_SECTIONS]

        # RRF fusion
        fused = vector_store.rrf_fusion(def_chunks, kw_filtered, top_k=5)

        if not fused:
            continue

        context = _build_context_block(fused)

        # Get policy name for attribution
        uploaded = vector_store.get_policy_by_id(policy_id)
        policy_name = uploaded.get("user_label", "Policy") if uploaded else "Policy"

        result = llm.chat_json(
            EXPLAIN_TERM_SYSTEM,
            f"TERM TO EXPLAIN: {term}\n\nCONTEXT BLOCK:\n{context}",
            temperature=0.0,
        )

        if result.get("found"):
            result["policy_name"] = policy_name
            return result

    # No grounded explanation found in any recommended policy
    return _not_found(term)


CHAT_REPLY_SYSTEM = """You are a knowledgeable, friendly health insurance advisor in India.

RULES:
1. If a CONTEXT BLOCK is provided, base your answer on it — quote specific clauses or numbers where helpful.
2. If no context, use accurate general knowledge about Indian health insurance.
3. Keep your answer conversational, clear, and under 4 sentences.
4. Be specific: mention real numbers, waiting periods, or coverage terms when relevant.
5. End with one actionable tip or follow-up suggestion.
6. Do NOT make up policy names, prices, or terms not in the context.

Return ONLY valid JSON:
{
  "answer": "your conversational response here",
  "suggest_policies": true or false
}

suggest_policies=true only if the user's question is best answered by showing them actual policy recommendations."""


def get_chat_reply(question: str, session_policy_ids: list[str] = []) -> dict:
    """
    Answer a conversational/educational question in natural language.
    Searches available policy PDFs for relevant context first.
    """
    context_parts: list[str] = []

    for policy_id in session_policy_ids[:2]:
        try:
            query_emb = embedder.embed_text(question)
            chunks = vector_store.section_search(
                query_emb, policy_id,
                ["coverage", "conditions", "definitions", "limits", "waiting_periods"],
                top_k=3,
            )
            if chunks:
                uploaded = vector_store.get_policy_by_id(policy_id)
                name = uploaded.get("user_label", "Policy") if uploaded else "Policy"
                context_parts.append(f"[From {name}]\n{_build_context_block(chunks)}")
        except Exception:
            continue

    user_msg = f"USER QUESTION: {question}"
    if context_parts:
        user_msg = f"CONTEXT BLOCK:\n{'---'.join(context_parts)}\n\n{user_msg}"

    result = llm.chat_json(CHAT_REPLY_SYSTEM, user_msg)
    return {
        "answer": result.get("answer", "I'm here to help with health insurance questions. Could you tell me what you're looking for?"),
        "suggest_policies": result.get("suggest_policies", False),
    }


def _not_found(term: str) -> dict:
    return {
        "found": False,
        "explanation": (
            f"'{term}' could not be found in the specific policy documents shown. "
            "Try asking after getting policy recommendations, or consult the insurer directly."
        ),
        "example": None,
        "citation": None,
        "policy_name": None,
    }
