"""
Specialized reusable AI skills:
- HiddenConditionsDetector: 3-layer hybrid RAG to find implicit policy traps
- CoverageGapScanner: Identify missing coverage areas in a policy
- PolicyRanker: Score and rank catalog policies for a user profile
"""
from __future__ import annotations
from services import embedder, vector_store, llm


# ── Hidden Conditions Detector ───────────────────────────────────────────────

HIDDEN_CONDITIONS_SYSTEM = """You are a senior health insurance claims consultant acting on behalf of the policyholder.
You have access to actual policy wording clauses below (direct answer, definitions, exclusions, and conditions sections).

Your job: Find BOTH what is explicitly stated AND what is IMPLICITLY implied or hidden in the policy language.

Specifically look for these hidden traps:
- room_rent_trap: Room rent cap → proportional deduction of ALL associated charges (surgeon, ICU, medicines all cut proportionally)
- pre_auth_required: Pre-authorization requirement — if missed, claim denied even if procedure is covered
- proportional_deduction: Any clause that proportionally reduces total claim based on a sub-limit breach
- definition_trap: Key term (e.g. "Medically Necessary", "Hospitalization", "Pre-existing Disease") defined narrowly
- waiting_period: Specific illness waiting period or PED waiting period that may apply
- sub_limit: A cap on a specific treatment type even though hospitalization is broadly covered
- documentation: Specific documents required that are non-obvious or time-sensitive
- network_restriction: Non-network hospital co-pay or full exclusion

Return ONLY valid JSON in this exact format:
{
  "verdict": "COVERED | NOT_COVERED | PARTIALLY_COVERED | AMBIGUOUS",
  "practical_claimability": "GREEN | AMBER | RED",
  "confidence": 0-100,
  "plain_answer": "one clear sentence for a layperson",
  "conditions": ["list of explicit conditions that apply"],
  "hidden_conditions": [
    {
      "type": "room_rent_trap|pre_auth_required|proportional_deduction|definition_trap|waiting_period|sub_limit|documentation|network_restriction",
      "description": "plain English explanation of the hidden condition",
      "impact": "concrete impact on the actual claim payout or process"
    }
  ],
  "citations": [
    {"text": "exact quoted clause from policy", "page": 14, "section": "Exclusions"}
  ],
  "recommendation": "specific actionable next step for the policyholder"
}

CRITICAL RULES:
- Only report hidden_conditions where you found actual textual evidence in the provided clauses
- Do NOT hallucinate clauses that are not in the provided text
- If the policy text does not address the question, return AMBIGUOUS with confidence < 40
- GREEN = clearly covered, simple claim process
- AMBER = technically covered but conditions/traps make claiming difficult
- RED = not covered or likely to be denied"""


class HiddenConditionsDetector:
    """Performs 3-layer hybrid RAG and returns structured verdict with hidden conditions."""

    def detect(self, question: str, policy_id: str) -> dict:
        # Embed the question once
        query_emb = embedder.embed_text(question)

        # Layer 1: Hybrid search — semantic + keyword → RRF fusion
        semantic = vector_store.semantic_search(query_emb, policy_id, top_k=8)
        keyword = vector_store.keyword_search(question, policy_id, top_k=8)
        fused = vector_store.rrf_fusion(semantic, keyword, top_k=5)

        # Layer 2: Definitions section
        definitions = vector_store.section_search(
            query_emb, policy_id, ["definitions"], top_k=3
        )

        # Layer 3: Exclusions + Conditions + Limits sections
        exclusions = vector_store.section_search(
            query_emb, policy_id, ["exclusions", "conditions", "limits", "waiting_periods"], top_k=3
        )

        # Build context for LLM
        def format_chunks(chunks: list[dict], label: str) -> str:
            if not chunks:
                return ""
            parts = [f"[{label}]"]
            for c in chunks:
                parts.append(
                    f"[Page {c.get('page_number', '?')} | {c.get('section_type', 'general')}]\n{c['content']}"
                )
            return "\n\n".join(parts)

        context = "\n\n---\n\n".join(filter(None, [
            format_chunks(fused, "DIRECT ANSWER CLAUSES"),
            format_chunks(definitions, "DEFINITIONS"),
            format_chunks(exclusions, "EXCLUSIONS & CONDITIONS"),
        ]))

        user_prompt = f"""QUESTION: {question}

POLICY CLAUSES:
{context}

Analyze the above policy clauses and return the JSON verdict."""

        result = llm.chat_json(HIDDEN_CONDITIONS_SYSTEM, user_prompt)

        # Fallback defaults
        result.setdefault("verdict", "AMBIGUOUS")
        result.setdefault("practical_claimability", "AMBER")
        result.setdefault("confidence", 30)
        result.setdefault("plain_answer", "Unable to determine coverage from the available policy text.")
        result.setdefault("conditions", [])
        result.setdefault("hidden_conditions", [])
        result.setdefault("citations", [])
        result.setdefault("recommendation", "Contact your insurer directly for clarification.")

        return result


# ── Coverage Gap Scanner ─────────────────────────────────────────────────────

COVERAGE_CHECKLIST = [
    ("maternity", "covers_maternity", "Maternity benefit", "HIGH",
     "Maternity hospitalization is a common need. Without this, deliveries are fully out-of-pocket."),
    ("opd", "covers_opd", "OPD (outpatient) coverage", "MEDIUM",
     "Regular doctor visits and prescriptions are not covered. Adds significant annual expense."),
    ("mental_health", "covers_mental_health", "Mental health coverage", "MEDIUM",
     "Psychiatric treatment not covered. IRDAI mandates this but many policies have sub-limits."),
    ("ayush", "covers_ayush", "AYUSH (Ayurveda, Yoga, Unani, Siddha, Homeopathy)", "LOW",
     "Alternative medicine treatments not covered."),
    ("dental", "covers_dental", "Dental treatment", "LOW",
     "Dental procedures (except accident-related) not covered."),
    ("restoration", "restoration_benefit", "Sum insured restoration", "HIGH",
     "If SI is exhausted mid-year, no coverage remains for the rest of the year."),
    ("ncb", "ncb_percent", "No Claim Bonus", "MEDIUM",
     "Policy does not reward claim-free years with coverage increase."),
]


class CoverageGapScanner:
    """Identifies coverage gaps in a catalog policy by comparing metadata against checklist."""

    def scan(self, catalog_policy: dict) -> list[dict]:
        gaps = []
        for feature_key, field, label, severity, description in COVERAGE_CHECKLIST:
            value = catalog_policy.get(field)
            is_missing = (value is False) or (value is None) or (value == 0)
            if is_missing:
                gaps.append({
                    "feature": feature_key,
                    "label": label,
                    "severity": severity,
                    "description": description,
                    "recommendation": f"Consider adding {label} as a rider or switching to a plan that includes it.",
                })

        # Check waiting periods
        ped_years = catalog_policy.get("waiting_period_preexisting_years", 0)
        if ped_years and ped_years >= 4:
            gaps.append({
                "feature": "long_ped_wait",
                "label": "Very long pre-existing disease waiting period",
                "severity": "HIGH",
                "description": f"Pre-existing conditions have a {ped_years}-year waiting period. Any known conditions won't be covered for {ped_years} years.",
                "recommendation": "Look for policies with reduced PED waiting period (2 years) or portability options.",
            })

        # Check room rent
        room_rent = catalog_policy.get("room_rent_limit", "")
        if room_rent and "%" in room_rent:
            gaps.append({
                "feature": "room_rent_cap",
                "label": "Room rent cap (proportional deduction risk)",
                "severity": "HIGH",
                "description": f"Room rent is capped at {room_rent}. If you choose a higher-category room, ALL charges (surgeon, ICU, nursing) are proportionally reduced.",
                "recommendation": "Choose a room within the policy limit, or upgrade to a plan with no room rent restriction.",
            })

        # Check co-pay
        co_pay = catalog_policy.get("co_pay_percent", 0)
        if co_pay and co_pay > 0:
            gaps.append({
                "feature": "co_pay",
                "label": f"Co-payment of {co_pay}%",
                "severity": "MEDIUM",
                "description": f"You pay {co_pay}% of every claim out-of-pocket. On a ₹5L claim, that's ₹{co_pay * 5000:,}.",
                "recommendation": "Consider a plan with 0% co-pay unless the premium saving justifies the risk.",
            })

        return gaps


# ── Policy Ranker ────────────────────────────────────────────────────────────

def hard_filter(policies: list[dict], req: dict) -> list[dict]:
    """
    Strict pre-filter before scoring. No LLM. No fallback.
    Policies that fail ANY hard criterion are excluded entirely.

    Hard criteria:
    - premium_min > budget_max → excluded (budget is a hard cap)
    - maternity required but policy doesn't cover → excluded
    - opd required but policy doesn't cover → excluded
    - mental_health required but policy doesn't cover → excluded
    - preferred_type set but doesn't match → excluded
    """
    out = []
    budget = req.get("budget_max")
    needs = req.get("needs", [])
    ptype = req.get("preferred_type")

    for p in policies:
        if budget and p.get("premium_min", 0) > budget:
            continue
        if "maternity" in needs and not p.get("covers_maternity"):
            continue
        if "opd" in needs and not p.get("covers_opd"):
            continue
        if "mental_health" in needs and not p.get("covers_mental_health"):
            continue
        if ptype and p.get("type") != ptype:
            continue
        out.append(p)
    return out


def _coverage_strength(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 45:
        return "medium"
    return "low"


def _estimated_waiting(policy: dict, req: dict) -> str:
    parts = []
    ped = policy.get("waiting_period_preexisting_years")
    if ped:
        parts.append(f"{ped} years for pre-existing conditions")
    mat_months = policy.get("waiting_period_maternity_months")
    needs = req.get("needs", [])
    if "maternity" in needs and mat_months:
        parts.append(f"{mat_months} months for maternity")
    general = policy.get("waiting_period_general", 30)
    parts.append(f"{general} days initial waiting period")
    return ", ".join(parts) if parts else "Not specified"


class PolicyRanker:
    """
    Deterministic policy ranking engine.

    Two-phase:
    1. hard_filter() — eliminates policies that fail hard constraints
    2. weighted_score() — scores survivors from 0 starting point
    """

    def rank(self, requirements: dict, policies: list[dict]) -> list[dict]:
        scored = []
        for policy in policies:
            score, why, tradeoffs = self._weighted_score(requirements, policy)
            strength = _coverage_strength(score)
            scored.append({
                **policy,
                "match_score": score,
                # Legacy field kept for existing frontend compatibility
                "match_reasons": why,
                # New rich fields
                "why_matched": why,
                "tradeoffs": tradeoffs,
                "estimated_waiting_period": _estimated_waiting(policy, requirements),
                "coverage_strength": strength,
            })
        return sorted(scored, key=lambda x: x["match_score"], reverse=True)

    def _weighted_score(self, req: dict, policy: dict) -> tuple[int, list[str], list[str]]:
        """
        Score starts at 0. Each factor adds or subtracts.
        Clamped to [0, 100].

        Coverage needs (when explicitly requested, these score HEAVILY):
          +30  maternity covered
          +20  OPD covered / -12 if not
          +15  mental_health covered / -8 if not
          +12  AYUSH covered / -8 if not
          +12  dental covered / -8 if not
          +12  NCB >= 50% / +6 if any NCB
          +15  restoration explicitly requested + covered / -10 if not
          + 5  restoration passively present (not requested)

        Pre-existing conditions:
          +25  NOT in exclusion list
          -20  explicitly in exclusion list

        Budget fit:
          +20  premium_min <= budget
          + 5  premium_min <= budget × 0.65 (exceptional value)

        Members / plan type fit:
          + 8  family_floater AND members >= 3
          + 5  individual AND members == 1

        Sum insured:
          +10  sum_insured_max >= sum_insured_min requirement
          -15  sum_insured_max < requirement

        Network:
          +10  network_hospitals > 5000

        Waiting period:
          +10  PED wait <= 2 years
          -15  PED wait >= 4 years

        Penalties:
          -10  co_pay > 0
          -10  room_rent_limit contains "%" (proportional deduction risk)
        """
        score = 0
        why: list[str] = []
        tradeoffs: list[str] = []
        needs = req.get("needs", [])
        budget = req.get("budget_max")
        preexisting = req.get("preexisting_conditions") or []
        exclusions = [e.lower() for e in (policy.get("exclusions") or [])]
        members = req.get("members")
        si_min = req.get("sum_insured_min")

        # ── Coverage needs (heavily weighted when explicitly requested) ──────────

        if "maternity" in needs:
            if policy.get("covers_maternity"):
                score += 30
                why.append("Covers maternity")
            else:
                tradeoffs.append("Maternity not covered")

        if "opd" in needs:
            if policy.get("covers_opd"):
                score += 20
                why.append("OPD coverage included")
            else:
                score -= 12
                tradeoffs.append("OPD not covered")

        if "mental_health" in needs:
            if policy.get("covers_mental_health"):
                score += 15
                why.append("Mental health coverage included")
            else:
                score -= 8
                tradeoffs.append("Mental health not covered")

        if "ayush" in needs:
            if policy.get("covers_ayush"):
                score += 12
                why.append("AYUSH/alternative medicine covered")
            else:
                score -= 8
                tradeoffs.append("AYUSH not covered")

        if "dental" in needs:
            if policy.get("covers_dental"):
                score += 12
                why.append("Dental coverage included")
            else:
                score -= 8
                tradeoffs.append("Dental not covered")

        if "ncb" in needs:
            ncb = policy.get("ncb_percent", 0)
            if ncb >= 50:
                score += 12
                why.append(f"{ncb}% No Claim Bonus — great long-term value")
            elif ncb > 0:
                score += 6
                why.append(f"{ncb}% No Claim Bonus")
            else:
                score -= 5
                tradeoffs.append("No NCB benefit")

        if "restoration" in needs:
            if policy.get("restoration_benefit"):
                score += 15
                why.append("Sum insured restoration benefit")
            else:
                score -= 10
                tradeoffs.append("No restoration benefit")
        elif policy.get("restoration_benefit"):
            # Passively good feature even if not requested
            score += 5
            why.append("Sum insured restored after claim")

        # ── Pre-existing conditions ──────────────────────────────────────────────

        if preexisting:
            excluded_any = any(
                any(word in excl for word in cond.lower().split() if len(word) > 3)
                for cond in preexisting
                for excl in exclusions
            )
            if not excluded_any:
                score += 25
                why.append("Pre-existing conditions not in exclusion list")
            else:
                score -= 20
                hit = next(
                    (c for c in preexisting if any(
                        any(w in excl for w in c.lower().split() if len(w) > 3)
                        for excl in exclusions
                    )), preexisting[0]
                )
                tradeoffs.append(f"'{hit}' may be excluded — verify policy wording")

        # ── Budget fit ───────────────────────────────────────────────────────────

        if budget:
            prem_min = policy.get("premium_min", 0)
            if prem_min <= budget:
                score += 20
                why.append(f"Premium from ₹{prem_min:,}/yr (within ₹{budget:,} budget)")
                if prem_min <= budget * 0.65:
                    score += 5
                    why.append("Excellent value for your budget")
            else:
                tradeoffs.append(f"Lowest premium ₹{prem_min:,}/yr exceeds budget")

        # ── Members / plan type fit ──────────────────────────────────────────────

        if members:
            try:
                m = int(members)
                if m >= 3 and policy.get("type") == "family_floater":
                    score += 8
                    why.append("Family floater — covers your whole family under one premium")
                elif m == 1 and policy.get("type") == "individual":
                    score += 5
                    why.append("Individual plan — right fit for single coverage")
            except (ValueError, TypeError):
                pass

        # ── Sum insured requirement ──────────────────────────────────────────────

        if si_min:
            si_max = policy.get("sum_insured_max", 0)
            if si_max >= si_min:
                score += 10
                why.append(f"Sum insured up to ₹{si_max // 100000:.0f}L available")
            else:
                score -= 15
                tradeoffs.append(f"Max sum insured ₹{si_max // 100000:.0f}L may not meet your ₹{si_min // 100000:.0f}L requirement")

        # ── Network hospitals ────────────────────────────────────────────────────

        network = policy.get("network_hospitals", 0)
        if network > 5000:
            score += 10
            why.append(f"{network:,} network hospitals")

        # ── Waiting period ───────────────────────────────────────────────────────

        ped = policy.get("waiting_period_preexisting_years", 4)
        if ped <= 2:
            score += 10
            why.append(f"Short {ped}-year PED waiting period")
        elif ped >= 4:
            score -= 15
            tradeoffs.append(f"Long {ped}-year pre-existing waiting period")

        # ── Penalties ────────────────────────────────────────────────────────────

        copay = policy.get("co_pay_percent", 0)
        if copay and copay > 0:
            score -= 10
            tradeoffs.append(f"{copay}% co-payment on every claim")

        room = policy.get("room_rent_limit") or ""
        if room and "%" in room:
            score -= 10
            tradeoffs.append(f"Room rent capped at {room} — proportional deduction applies")

        return max(0, min(100, score)), why, tradeoffs
