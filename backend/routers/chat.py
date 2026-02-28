"""
Chat Memory System — persistent conversational sessions stored in Supabase.

Endpoints:
  POST   /api/chat/sessions                      — create new session
  GET    /api/chat/sessions                      — list recent sessions
  GET    /api/chat/sessions/{session_id}         — get session + all messages
  POST   /api/chat/sessions/{session_id}/messages — send message + get AI response
  DELETE /api/chat/sessions/{session_id}         — delete session (cascades messages)

AI response logic:
  - Retrieve last 10 messages from DB for context
  - If context > 6000 chars: summarize oldest portion with LLM
  - Route through discover_chat logic (follow-up questions or ranked results)
  - Persist both user message and assistant response
  - Update session.updated_at and session.context with any extracted state
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services import llm
from services.vector_store import get_client
from services.skills import PolicyRanker, hard_filter
from services.vector_store import list_catalog_policies
from services.advisor_agent import (
    classify_intent,
    find_uploaded_for_insurer,
    get_rag_insights,
    explain_term,
    get_chat_reply,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])
ranker = PolicyRanker()

CONTEXT_SUMMARY_SYSTEM = """Summarize this insurance advisor conversation in 3-4 sentences.
Capture: what coverage the user needs, their budget, family size, and any pre-existing conditions mentioned.
Return ONLY the summary text, no JSON."""

CHAT_INTRO_SYSTEM = """You are a warm health insurance advisor. Write a friendly 1-2 sentence response acknowledging what the user asked for, right before showing their policy recommendations. Be specific about what you understood. Do not say "Great!" or "Sure!" — be natural.
Return ONLY valid JSON: {"message": "your response here"}"""

NO_RESULTS_MESSAGE = (
    "No policies in our catalog match all your hard requirements. "
    "Try relaxing your budget, removing a specific coverage requirement, "
    "or changing the plan type."
)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db():
    return get_client()


def _create_session(user_id: str = "anonymous", session_name: Optional[str] = None) -> dict:
    data = {"user_id": user_id, "context": {}}
    if session_name:
        data["session_name"] = session_name
    res = _db().table("chat_sessions").insert(data).execute()
    return res.data[0]


def _get_session(session_id: str) -> dict | None:
    res = _db().table("chat_sessions").select("*").eq("id", session_id).execute()
    return res.data[0] if res.data else None


def _list_sessions(limit: int = 20) -> list[dict]:
    res = (
        _db().table("chat_sessions")
        .select("id, user_id, session_name, context, created_at, updated_at")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def _get_messages(session_id: str, limit: int = 100) -> list[dict]:
    res = (
        _db().table("chat_messages")
        .select("id, role, content, metadata, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return res.data or []


def _insert_message(session_id: str, role: str, content: str, metadata: dict | None = None) -> dict:
    row = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }
    res = _db().table("chat_messages").insert(row).execute()
    return res.data[0]


def _update_session(session_id: str, context: dict):
    from datetime import datetime, timezone
    _db().table("chat_sessions").update({
        "context": context,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", session_id).execute()


def _delete_session(session_id: str):
    _db().table("chat_sessions").delete().eq("id", session_id).execute()


# ── Context management ────────────────────────────────────────────────────────

def _build_context_string(messages: list[dict]) -> str:
    """Build conversation string from last 10 messages."""
    recent = messages[-10:] if len(messages) > 10 else messages
    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])


def _maybe_summarize(full_context: str) -> str:
    """If context is too long, summarize the older half to keep token count manageable."""
    if len(full_context) <= 6000:
        return full_context
    midpoint = len(full_context) // 2
    old_half = full_context[:midpoint]
    recent_half = full_context[midpoint:]
    summary = llm.chat_text(CONTEXT_SUMMARY_SYSTEM, old_half, temperature=0.1)
    return f"[EARLIER CONVERSATION SUMMARY]\n{summary}\n\n[RECENT MESSAGES]\n{recent_half}"


def _process_message(content: str, db_messages: list[dict], session_context: dict) -> dict:
    """
    3-mode conversational advisor (mirrors discovery.py /discover/chat logic).
      GATHER  — asks smart follow-up questions until all 3 essential fields present
      EXPLAIN — explains insurance terms grounded in actual uploaded PDF text
      RECOMMEND — hard filter + weighted rank + RAG insights from PDF for top 3
    """
    context_str = _build_context_string(db_messages)
    context_str = _maybe_summarize(context_str)

    # Classify intent and extract requirements from full conversation
    intent_result = classify_intent(context_str)
    intent = intent_result.get("intent", "gather_info")
    extracted = intent_result.get("extracted") or {}
    extracted["needs"] = extracted.get("needs") or []
    extracted["preexisting_conditions"] = extracted.get("preexisting_conditions") or []

    # MODE GATHER: any essential field (budget / members / needs) missing
    missing_any = (
        not intent_result.get("has_budget")
        or not intent_result.get("has_members")
        or not intent_result.get("has_needs_or_conditions")
    )
    if intent == "gather_info" or (missing_any and intent not in ("explain_term", "explain_policy", "chat_reply")):
        question = (
            intent_result.get("next_question")
            or "Could you tell me your health coverage needs, annual budget, and family size?"
        )
        return {"type": "question", "message": question}

    # MODE CHAT: conversational / educational reply
    if intent == "chat_reply":
        session_policy_ids = session_context.get("last_recommended_uploaded_ids", [])
        reply = get_chat_reply(content, session_policy_ids)
        return {"type": "chat", "message": reply["answer"]}

    # MODE EXPLAIN: user asked about an insurance term or specific policy
    if intent in ("explain_term", "explain_policy"):
        term = intent_result.get("term_to_explain") or intent_result.get("policy_name_asked")
        if term:
            # Retrieve uploaded policy IDs stored in session context from last recommendation
            session_policy_ids = session_context.get("last_recommended_uploaded_ids", [])
            result = explain_term(term, session_policy_ids)
            return {
                "type": "explanation",
                "message": result.get("explanation", ""),
                "example": result.get("example"),
                "citation": result.get("citation"),
                "policy_name": result.get("policy_name"),
                "found": result.get("found", False),
            }

    # MODE RECOMMEND: all 3 essential fields present
    all_policies = list_catalog_policies()
    filtered = hard_filter(all_policies, extracted)

    if not filtered:
        return {
            "type": "no_results",
            "message": NO_RESULTS_MESSAGE,
            "extracted_requirements": extracted,
            "policies": [],
            "total_found": 0,
        }

    ranked = ranker.rank(extracted, filtered)
    top_policies = ranked[:6]
    user_needs = extracted["needs"] + extracted["preexisting_conditions"]

    # RAG enrichment: top 3 policies → find matching uploaded PDF → surface hidden traps
    uploaded_ids: list[str] = []
    for policy in top_policies[:3]:
        uploaded = find_uploaded_for_insurer(policy.get("insurer", ""))
        if uploaded:
            insights = get_rag_insights(uploaded["id"], user_needs)
            policy["rag_insights"] = insights
            policy["uploaded_policy_id"] = uploaded["id"]
            uploaded_ids.append(uploaded["id"])
        else:
            policy["rag_insights"] = {"available": False}

    last_user = next(
        (m["content"] for m in reversed(db_messages) if m["role"] == "user"), content
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
        "uploaded_policy_ids": uploaded_ids,
    }


# ── Request models ────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = "anonymous"
    session_name: Optional[str] = None


class SendMessageRequest(BaseModel):
    content: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/sessions")
async def create_session(req: CreateSessionRequest):
    """Create a new chat session. Returns session_id for client to store."""
    session = _create_session(req.user_id or "anonymous", req.session_name)
    return {
        "session_id": session["id"],
        "created_at": session["created_at"],
    }


@router.get("/sessions")
async def list_sessions():
    """List the 20 most recent sessions ordered by last activity."""
    sessions = _list_sessions(limit=20)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session metadata + full message history."""
    session = _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = _get_messages(session_id)
    return {
        "session": session,
        "messages": messages,
    }


@router.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, req: SendMessageRequest):
    """
    Process a user message:
    1. Persist user message
    2. Load last 10 messages for context
    3. Generate AI response (follow-up or ranked policies)
    4. Persist assistant response
    5. Update session context with extracted state
    6. Return AI response
    """
    session = _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Persist user message
    _insert_message(session_id, "user", req.content)

    # Get all messages for context (last 10 used internally)
    db_messages = _get_messages(session_id)

    # Generate AI response
    ai_response = _process_message(req.content, db_messages, session.get("context", {}))

    # Persist assistant message with metadata
    metadata = {
        "type": ai_response.get("type"),
        "policies": ai_response.get("policies", []),
        "extracted_requirements": ai_response.get("extracted_requirements", {}),
    }
    persisted = _insert_message(session_id, "assistant", ai_response["message"], metadata)

    # Update session context with extracted state + uploaded policy IDs for term lookups
    updated_context = {**session.get("context", {})}
    extracted = ai_response.get("extracted_requirements", {})
    if extracted:
        if extracted.get("budget_max"):
            updated_context["budget"] = extracted["budget_max"]
        if extracted.get("preexisting_conditions"):
            updated_context["diseases"] = extracted["preexisting_conditions"]
        if extracted.get("members"):
            updated_context["family_size"] = extracted["members"]
    # Store uploaded PDF IDs so future explain_term calls can look up the right documents
    if ai_response.get("uploaded_policy_ids"):
        updated_context["last_recommended_uploaded_ids"] = ai_response["uploaded_policy_ids"]
    if updated_context != session.get("context", {}):
        _update_session(session_id, updated_context)

    return {
        **ai_response,
        "message_id": persisted["id"],
        "session_id": session_id,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete session and all its messages (cascade)."""
    session = _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    _delete_session(session_id)
    return {"deleted": True, "session_id": session_id}
