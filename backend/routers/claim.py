"""
Claim Advisory, Medical Matching, and Coverage Gap Analysis routes.
Feature 4: Medical report → extract conditions → match against policies
Feature 5: Existing policy + diagnosis → deterministic claim eligibility
Feature 6: Coverage gap analysis for any catalog policy
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from services import vector_store
from services.claim_engine import run_claim_check
from services.skills import HiddenConditionsDetector, CoverageGapScanner
from services.medical_extractor import extract_from_text, extract_from_pdf_bytes, match_conditions_to_exclusions
from services.advisor_agent import find_uploaded_for_insurer, get_rag_insights

router = APIRouter(prefix="/api", tags=["claim"])
gap_scanner = CoverageGapScanner()
detector = HiddenConditionsDetector()  # kept for gap-analysis uploaded policies


class ClaimCheckRequest(BaseModel):
    policy_id: str
    diagnosis: str
    treatment_type: Optional[str] = "hospitalization"


class ExtractConditionsRequest(BaseModel):
    text: str


class MatchConditionsRequest(BaseModel):
    conditions: list[dict]


# ── Feature 5: Claim Advisory (Deterministic RAG Engine) ────────────────────

@router.post("/claim-check")
async def claim_check(req: ClaimCheckRequest):
    """
    Deterministic claim eligibility check.
    RAG retrieves only relevant clauses from the policy document.
    Score is computed by rule-based function — not by LLM.
    """
    # Verify policy exists (uploaded or catalog)
    policy = vector_store.get_policy_by_id(req.policy_id)
    catalog = vector_store.get_catalog_policy(req.policy_id)
    if not policy and not catalog:
        raise HTTPException(status_code=404, detail="Policy not found.")

    result = run_claim_check(
        policy_id=req.policy_id,
        condition=req.diagnosis,
        treatment_type=req.treatment_type or "hospitalization",
    )

    # If no relevant chunks found — return 422 with structured error
    if result.get("error"):
        return JSONResponse(
            status_code=422,
            content={"detail": result["error"]},
        )

    return result


# ── Feature 4: Medical Report → Policy Matching ──────────────────────────────

@router.post("/extract-conditions")
async def extract_conditions_from_text_endpoint(req: ExtractConditionsRequest):
    """Extract medical conditions from free text input."""
    return extract_from_text(req.text)


@router.post("/extract-conditions-file")
async def extract_conditions_from_file(file: UploadFile = File(...)):
    """Extract medical conditions from uploaded medical report PDF."""
    contents = await file.read()
    return extract_from_pdf_bytes(contents)


@router.post("/match-conditions")
async def match_conditions(req: MatchConditionsRequest):
    """
    Given extracted conditions, rank all catalog policies by suitability.
    Flags policies where conditions may be excluded.
    """
    all_policies = vector_store.list_catalog_policies()
    flagged = match_conditions_to_exclusions(req.conditions, all_policies)

    # Sort: fewer exclusion flags first
    flagged.sort(key=lambda p: len(p.get("exclusion_flags", [])))

    return {
        "extracted_conditions": req.conditions,
        "recommended_policies": flagged[:6],
        "total_evaluated": len(flagged),
    }


# ── Feature 6: Coverage Gap Analysis ────────────────────────────────────────

@router.get("/gap-analysis/{policy_id}")
async def gap_analysis(policy_id: str):
    """
    Identify coverage gaps in a policy.
    Catalog policies: rule-based metadata scan.
    Uploaded policies: RAG-based analysis from chunks.
    """
    catalog_policy = vector_store.get_catalog_policy(policy_id)
    if catalog_policy:
        gaps = gap_scanner.scan(catalog_policy)
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        gaps.sort(key=lambda g: severity_order.get(g["severity"], 3))

        # RAG enrichment: find the uploaded PDF for this insurer → surface hidden conditions
        rag_hidden: list[dict] = []
        rag_available = False
        uploaded = find_uploaded_for_insurer(catalog_policy.get("insurer", ""))
        if uploaded:
            rag_available = True
            gap_needs = [
                "room rent", "co-pay", "waiting period", "sub-limit",
                "exclusion", "pre-authorization", "proportional deduction",
                "co-payment", "deductible", "network hospital",
            ]
            insights = get_rag_insights(uploaded["id"], gap_needs)
            if insights.get("available") and insights.get("hidden_traps"):
                rag_hidden = insights["hidden_traps"]

        return {
            "policy_name": catalog_policy.get("name"),
            "insurer": catalog_policy.get("insurer"),
            "analysis_type": "catalog_based",
            "gaps": gaps,
            "gap_count": len(gaps),
            "high_risk_count": sum(1 for g in gaps if g["severity"] == "HIGH"),
            "hidden_conditions": rag_hidden,
            "rag_available": rag_available,
        }

    uploaded = vector_store.get_policy_by_id(policy_id)
    if not uploaded:
        raise HTTPException(status_code=404, detail="Policy not found.")

    gap_question = (
        "What coverage gaps does this policy have? "
        "Does it lack maternity, OPD, mental health, dental, restoration, or NCB benefits? "
        "Are there any high waiting periods, room rent caps, or co-pay requirements?"
    )
    gap_result = detector.detect(gap_question, policy_id)

    return {
        "policy_name": uploaded.get("user_label", "Unknown Policy"),
        "analysis_type": "rag_based",
        "gaps": [],
        "ai_summary": gap_result.get("plain_answer"),
        "hidden_conditions": gap_result.get("hidden_conditions", []),
        "recommendation": gap_result.get("recommendation"),
    }
