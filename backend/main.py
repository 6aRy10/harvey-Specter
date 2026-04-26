"""
LexAgents API — FastAPI Backend
================================
Main API server for the AI Law Firm.

Run with: uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from backend.agents.orchestrator import Orchestrator

# ─── App Setup ───
app = FastAPI(
    title="LexAgents — AI Law Firm API",
    description="Autonomous AI legal department with specialized agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Initialize ───
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
orchestrator = Orchestrator(openai_client)


# ─── Request Models ───
class MatterRequest(BaseModel):
    request: str
    contract_text: str | None = None


class ReviewRequest(BaseModel):
    contract_text: str
    jurisdiction: str | None = None
    context: dict | None = None  # Smart intake answers: goal, deadline, bestcase, worstcase, etc.


class DraftRequest(BaseModel):
    requirements: str
    jurisdiction: str | None = None


class ResearchRequest(BaseModel):
    question: str
    jurisdiction: str | None = None


class ModifyRequest(BaseModel):
    contract_text: str
    modifications: str


# ─── Health Check ───
@app.get("/")
async def root():
    return {
        "name": "LexAgents — AI Law Firm",
        "version": "1.0.0",
        "agents": ["intake", "contract_review", "legal_research", "drafting", "orchestrator"],
        "status": "operational",
    }


# ─── Full Matter Processing (end-to-end) ───
@app.post("/api/matters")
async def create_matter(req: MatterRequest):
    """
    Create a new legal matter. The orchestrator will:
    1. Classify the request (Intake Agent)
    2. Route to appropriate specialist agents
    3. Return consolidated results with audit trail
    """
    try:
        result = await orchestrator.create_matter(req.request, req.contract_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/matters")
async def list_matters():
    """List all processed matters."""
    return orchestrator.list_matters()


# ─── Settings ───
class SlackSettingsRequest(BaseModel):
    webhook_url: str


@app.get("/api/settings")
async def get_settings():
    """Return current runtime configuration status."""
    from backend.integrations.slack_approval import is_configured, get_webhook_url
    wh = get_webhook_url() or ""
    masked = (wh[:30] + "...") if len(wh) > 30 else wh
    return {
        "slack_configured": is_configured(),
        "slack_webhook_preview": masked if wh else "",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/api/settings/slack-webhook")
async def set_slack_webhook(req: SlackSettingsRequest):
    """Set primary Slack webhook URL at runtime."""
    url = req.webhook_url.strip()
    if not url.startswith("https://hooks.slack.com/"):
        raise HTTPException(status_code=400, detail="Must be a valid Slack webhook URL (https://hooks.slack.com/...)")
    os.environ["SLACK_WEBHOOK_URL"] = url
    env_path = Path(__file__).parent.parent / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        new_lines = [l for l in lines if not l.startswith("SLACK_WEBHOOK_URL=")]
        new_lines.append(f"SLACK_WEBHOOK_URL={url}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception:
        pass
    return {"status": "ok", "slack_configured": True, "message": "Slack webhook saved."}


# In-memory multi-target store  { name: {"url": webhook_url, "role": "legal_team"|"executive"|"general"} }
_slack_targets: dict = {}

# Pipeline result store  { matter_id: full_result_dict }
_pipeline_store: dict = {}


def _load_slack_targets_from_env():
    """Load slack targets from OS env vars first, then .env file as fallback."""
    import re
    # ── 1. Read from OS environment variables (works on Render/Netlify/any host) ──
    for key, val in os.environ.items():
        m = re.match(r'^SLACK_TARGET_(.+?)_URL$', key)
        if m:
            raw_name = m.group(1)
            url = val.strip()
            name = raw_name.replace("_", " ").title()
            role = os.getenv(f"SLACK_TARGET_{raw_name}_ROLE", "general").strip()
            if url:
                _slack_targets[name] = {"url": url, "role": role}
    # ── 2. Also read .env file (local dev fallback) ──
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^SLACK_TARGET_(.+?)_URL=(.+)$', line.strip())
        if m:
            raw_name, url = m.group(1), m.group(2).strip()
            name = raw_name.replace("_", " ").title()
            if name not in _slack_targets:  # don't overwrite OS env vars
                role_line = next(
                    (l for l in env_path.read_text(encoding="utf-8").splitlines()
                     if l.startswith(f"SLACK_TARGET_{raw_name}_ROLE=")), None)
                role = role_line.split("=", 1)[1].strip() if role_line else "general"
                _slack_targets[name] = {"url": url, "role": role}


_load_slack_targets_from_env()


class SlackTargetRequest(BaseModel):
    name: str
    webhook_url: str
    role: str = "general"  # "legal_team" | "executive" | "general"


@app.get("/api/settings/slack-targets")
async def list_slack_targets():
    """List all named Slack webhook targets."""
    targets = [
        {"name": k, "webhook_preview": v["url"][:40] + "..." if len(v["url"]) > 40 else v["url"], "role": v.get("role", "general")}
        for k, v in _slack_targets.items()
    ]
    primary = os.getenv("SLACK_WEBHOOK_URL", "")
    if primary and "Primary" not in _slack_targets:
        targets = [{"name": "Primary", "webhook_preview": primary[:40] + "...", "role": "general"}] + targets
    return {"targets": targets}


@app.post("/api/settings/slack-targets")
async def add_slack_target(req: SlackTargetRequest):
    """Add a named Slack webhook target with a role."""
    url = req.webhook_url.strip()
    name = req.name.strip()
    role = req.role.strip() or "general"
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not url.startswith("https://hooks.slack.com/"):
        raise HTTPException(status_code=400, detail="Must be a valid Slack webhook URL")
    _slack_targets[name] = {"url": url, "role": role}
    if not os.getenv("SLACK_WEBHOOK_URL"):
        os.environ["SLACK_WEBHOOK_URL"] = url
    env_path = Path(__file__).parent.parent / ".env"
    try:
        raw_key = name.upper().replace(" ", "_")
        url_key = f"SLACK_TARGET_{raw_key}_URL"
        role_key = f"SLACK_TARGET_{raw_key}_ROLE"
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        new_lines = [l for l in lines if not l.startswith(f"{url_key}=") and not l.startswith(f"{role_key}=")]
        new_lines.append(f"{url_key}={url}")
        new_lines.append(f"{role_key}={role}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception:
        pass
    return {"status": "ok", "name": name, "role": role, "targets_count": len(_slack_targets)}


@app.delete("/api/settings/slack-targets/{name}")
async def remove_slack_target(name: str):
    """Remove a named Slack target."""
    if name in _slack_targets:
        del _slack_targets[name]
    return {"status": "ok", "removed": name}


def _get_webhook_by_role(role: str) -> str:
    """Find first webhook URL matching a role."""
    for v in _slack_targets.values():
        if isinstance(v, dict) and v.get("role") == role:
            return v["url"]
    return ""


def _send_slack_raw(webhook_url: str, message: dict) -> bool:
    """Post a raw Slack block message. Returns True on success."""
    if not webhook_url:
        return False
    try:
        import requests as _req
        r = _req.post(webhook_url, json=message, headers={"Content-Type": "application/json"}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


class SlackTestRequest(BaseModel):
    webhook_url: str


@app.post("/api/settings/slack-test")
async def test_slack_webhook(req: SlackTestRequest):
    """Send a test message to verify a webhook works."""
    import requests as req_lib
    url = req.webhook_url.strip()
    if not url.startswith("https://hooks.slack.com/"):
        raise HTTPException(status_code=400, detail="Invalid Slack webhook URL")
    try:
        resp = req_lib.post(url, json={"text": "✅ Harveyy AI — Slack integration test successful! Pipeline notifications are now active."}, timeout=8)
        if resp.status_code == 200:
            return {"status": "ok", "message": "Test message sent successfully!"}
        else:
            raise HTTPException(status_code=400, detail=f"Slack returned {resp.status_code}: {resp.text}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """Extract text from PDF — text-based via pypdf, scanned via OpenAI Vision OCR."""
    try:
        import io, base64
        from pypdf import PdfReader
        contents = await file.read()

        # Step 1: Try pypdf (fast, works for text-based PDFs)
        reader = PdfReader(io.BytesIO(contents))
        text = "".join(p.extract_text() or "" for p in reader.pages).strip()
        if len(text) >= 200:
            return {"text": text, "pages": len(reader.pages), "chars": len(text), "method": "text"}

        # Step 2: Scanned PDF — OCR each page via OpenAI Vision (parallel)
        import fitz  # pymupdf
        doc = fitz.open(stream=contents, filetype="pdf")
        MAX_PAGES = 4
        pages_to_ocr = min(doc.page_count, MAX_PAGES)

        def ocr_page(i):
            pix = doc[i].get_pixmap(dpi=100)
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
                    {"type": "text", "text": "Extract all text from this contract page. Return only the text."}
                ]}],
                max_tokens=1000,
            )
            return resp.choices[0].message.content.strip()

        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, ocr_page, i) for i in range(pages_to_ocr)]
        all_text = await asyncio.gather(*tasks)
        extracted = "\n\n".join(all_text)
        return {"text": extracted, "pages": doc.page_count, "chars": len(extracted), "method": "ocr",
                "note": f"{pages_to_ocr}/{doc.page_count} pages"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/matters/{matter_id}")
async def get_matter(matter_id: str):
    """Get a specific matter by ID."""
    matter = orchestrator.get_matter(matter_id)
    if not matter:
        raise HTTPException(status_code=404, detail="Matter not found")
    return matter


# ─── Direct Agent Endpoints ───
@app.post("/api/review")
async def review_contract(req: ReviewRequest):
    """
    Full contract review pipeline:
    1. Contract Review Agent — flags risky clauses
    2. Legal Research Agent — auto-researches top flagged issues
    3. Suggests next steps (redraft, compliance check, etc.)
    """
    try:
        matter_id = f"REV-{uuid.uuid4().hex[:6].upper()}"
        audit = []

        # Truncate aggressively for fast response (< 10 seconds target)
        MAX_CHARS = 5000
        if len(req.contract_text) > MAX_CHARS:
            req.contract_text = req.contract_text[:MAX_CHARS] + "\n\n[... contract truncated — first 5,000 characters reviewed ...]"

        # Build context string from intake answers
        intake_context = ""
        if req.context:
            parts = []
            if req.context.get("goal"): parts.append(f"Client's objective: {req.context['goal']}")
            if req.context.get("role") or req.context.get("side"): parts.append(f"Client represents: {req.context.get('role') or req.context.get('side')}")
            if req.context.get("deadline"): parts.append(f"Deadline: {req.context['deadline']}")
            if req.context.get("concern"): parts.append(f"Specific concern: {req.context['concern']}")
            if req.context.get("bestcase"): parts.append(f"Best-case outcome: {req.context['bestcase']}")
            if req.context.get("worstcase"): parts.append(f"Worst-case to avoid: {req.context['worstcase']}")
            if req.context.get("transfers"): parts.append(f"Cross-border data transfers: {req.context['transfers']}")
            if req.context.get("jurisdiction"): parts.append(f"Jurisdiction: {req.context['jurisdiction']}")
            intake_context = "\n".join(parts)

        # Merge intake context into review context
        review_context = {"jurisdiction": req.jurisdiction} if req.jurisdiction else {}
        if intake_context:
            review_context["intake_brief"] = intake_context

        # Inject firm playbook/KB context into review prompt (Harvey-style)
        try:
            from data_pipeline.vector_store import search_vector_store
            contract_type_hint = req.context.get("type", "") if req.context else ""
            kb_hits = search_vector_store(
                f"{contract_type_hint} {req.contract_text[:300]}",
                "firm_policies", 3
            )
            if kb_hits:
                playbook_ctx = "\n".join(
                    f"- [{h.get('metadata',{}).get('source','Firm KB')}]: {h.get('text','')[:300]}"
                    for h in kb_hits
                )
                review_context["firm_playbook"] = (
                    "The following are your firm's standard policies/playbook clauses. "
                    "Flag any deviations from these standards:\n" + playbook_ctx
                )
        except Exception:
            pass

        # Step 1: Contract Review
        audit.append({"agent": "contract_review", "action": "review_started", "timestamp": datetime.now().isoformat()})
        if intake_context:
            audit.append({"agent": "intake", "action": "context_provided", "details": intake_context[:200]})
        review = await orchestrator.review_contract_only(req.contract_text, req.jurisdiction, review_context if review_context else None)
        audit.append({"agent": "contract_review", "action": "review_complete", "details": f"Risk: {review.get('overall_risk_score', 'N/A')}"})

        # Step 2: RAG search — runs in parallel with review result processing
        flagged = review.get("clauses_found", [])
        contract_type = review.get("contract_type", "contract")
        rag_insights = []
        async def run_rag_search():
            try:
                from data_pipeline.vector_store import search_vector_store
                query = f"{contract_type} {' '.join([c.get('clause_type','') for c in flagged[:3]])}"
                results = search_vector_store(query, "legal_knowledge", 4)
                return results or []
            except Exception:
                return []
        rag_task = asyncio.create_task(run_rag_search())

        research_results = []
        if False and flagged:  # auto-research disabled for fast mode
            top_issues = flagged[:3]
            focus = f" The client's goal is: {req.context.get('goal', 'full review')}." if req.context and req.context.get("goal") else ""

            async def research_clause(clause):
                question = f"Under German and EU law, what are the legal implications of a {clause.get('clause_type', 'clause')} clause that {clause.get('explanation', 'is potentially problematic')}?{focus}"
                res = await orchestrator.research_only(question, req.jurisdiction)
                return {
                    "clause_type": clause.get("clause_type"),
                    "risk_level": clause.get("risk_level"),
                    "research": res.get("legal_analysis", "")[:500],
                    "applicable_laws": res.get("applicable_laws", [])[:3],
                    "recommendation": (res.get("recommendations") or [""])[0] if res.get("recommendations") else "",
                }

            audit.append({"agent": "legal_research", "action": "parallel_research_started", "details": f"{len(top_issues)} clauses"})
            tasks = [research_clause(c) for c in top_issues]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, dict):
                    research_results.append(r)
            audit.append({"agent": "legal_research", "action": "parallel_research_complete", "details": f"{len(research_results)} succeeded"})

        # Step 3: Generate next-step suggestions based on findings
        suggested_actions = []
        risk_score = review.get("overall_risk_score", 0)
        if risk_score >= 7:
            suggested_actions.append({"action": "redraft_contract", "priority": "high", "reason": f"Risk score {risk_score}/10 — significant issues found. AI can auto-redraft with safer clauses."})
        if any(c.get("risk_level") == "HIGH" for c in flagged):
            suggested_actions.append({"action": "partner_review", "priority": "high", "reason": "High-risk clauses detected — requires partner sign-off before proceeding."})
        if review.get("missing_protections"):
            suggested_actions.append({"action": "add_protections", "priority": "medium", "reason": f"{len(review.get('missing_protections', []))} missing protections identified. AI can draft supplemental clauses."})
        if review.get("gdpr_compliance_notes"):
            suggested_actions.append({"action": "compliance_check", "priority": "medium", "reason": "GDPR implications detected — run full compliance audit."})
        suggested_actions.append({"action": "negotiate", "priority": "low", "reason": "Generate a redline version with suggested amendments for counterparty."})

        rag_insights = await rag_task

        return {
            "id": matter_id,
            "status": "complete",
            "agent": "contract_review",
            "agent_results": {"contract_review": review},
            "auto_research": research_results,
            "rag_insights": rag_insights,
            "suggested_actions": suggested_actions,
            "intake_brief": req.context if req.context else None,
            "audit_trail": audit,
            "agents_used": ["intake", "contract_review", "legal_research"] if (research_results and req.context) else ["contract_review", "legal_research"] if research_results else ["contract_review"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/draft")
async def draft_contract(req: DraftRequest):
    """Direct contract drafting (skip intake)."""
    try:
        result = await orchestrator.draft_contract_only(req.requirements, req.jurisdiction)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research")
async def legal_research(req: ResearchRequest):
    """Direct legal research (skip intake)."""
    try:
        result = await orchestrator.research_only(req.question, req.jurisdiction)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    message: str
    contract_text: str | None = None
    context: dict | None = None


@app.post("/api/chat")
async def chat_with_ai(req: ChatRequest):
    """
    Multilingual chat endpoint. Supports English, German (Deutsch), and Hindi.
    If a contract is loaded, answers questions about it with precise citations.
    Otherwise, does general German/EU law research with RAG.
    """
    try:
        msg = req.message.strip()

        # Truncate large contracts
        MAX_CHARS = 15000
        if req.contract_text and len(req.contract_text) > MAX_CHARS:
            req.contract_text = req.contract_text[:MAX_CHARS] + "\n\n[... contract truncated ...]"

        # If contract is loaded, do a context-aware review/research
        if req.contract_text and len(req.contract_text.strip()) > 50:
            # Use the contract review agent with the chat message as goal
            ctx = req.context or {}
            ctx["goal"] = msg
            review = await orchestrator.review_contract_only(req.contract_text, ctx.get("jurisdiction"), {"intake_brief": f"Client asks: {msg}"})
            summary = review.get("executive_summary", "")
            clauses = review.get("clauses_found", [])
            recs = review.get("top_recommendations", [])

            # Build a rich chat response
            response_parts = [summary] if summary else []
            if clauses:
                top = clauses[:3]
                for c in top:
                    response_parts.append(f"• {c.get('clause_type','')}: {c.get('explanation','')}")
            if recs:
                response_parts.append("\nRecommendations: " + "; ".join(recs[:3]))

            return {
                "reply": "\n".join(response_parts) if response_parts else "Review complete. See results panel for details.",
                "full_result": review,
                "sources": [],
                "detected_language": "auto",
            }
        else:
            # General legal research with RAG — multilingual
            research = await orchestrator.research_only(msg)
            lang = research.get("detected_language", "en")
            out_of_scope = research.get("out_of_scope", False)
            kb_hit = research.get("kb_hit", False)

            if out_of_scope:
                return {
                    "reply": "I'm a legal research tool — I can only help with questions about contracts, law, GDPR, compliance, and similar legal topics. That question is outside my scope.",
                    "full_result": research,
                    "sources": [],
                    "detected_language": lang,
                    "out_of_scope": True,
                    "kb_hit": False,
                }

            analysis = research.get("legal_analysis", "")
            sources = research.get("sources_cited", [])
            laws = research.get("applicable_laws", [])

            reply_parts = [analysis[:800]] if analysis else ["I couldn't find specific information on that topic."]
            if not kb_hit:
                reply_parts.append("\n⚠️ Note: This answer is based on general legal knowledge — not found in the firm knowledge base.")
            if laws:
                reply_parts.append("\n📖 Cited Laws:")
                for l in laws[:5]:
                    cite = l.get("exact_citation", l.get("section", ""))
                    reply_parts.append(f"  • {l.get('law','')} {cite} — {l.get('relevance','')}")
            if sources:
                reply_parts.append("\n📚 Sources: " + ", ".join(sources[:5]))

            return {
                "reply": "\n".join(reply_parts),
                "full_result": research,
                "sources": sources,
                "detected_language": lang,
                "out_of_scope": False,
                "kb_hit": kb_hit,
            }
    except Exception as e:
        return {"reply": f"Error: {str(e)}", "full_result": None, "sources": [], "detected_language": "en"}


@app.post("/api/modify")
async def modify_contract(req: ModifyRequest):
    """Modify an existing contract."""
    try:
        from backend.agents.drafting_agent import DraftingAgent
        drafter = DraftingAgent(openai_client)
        result = await drafter.modify_contract(req.contract_text, req.modifications)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── File Upload for Contract Review ───
@app.post("/api/review/upload")
async def review_uploaded_contract(
    file: UploadFile = File(...),
    jurisdiction: str = Form(None),
):
    """Upload a contract file (TXT, MD, or PDF) for review."""
    try:
        content = await file.read()

        if file.filename.endswith(".pdf"):
            # Extract text from PDF
            try:
                from pypdf import PdfReader
                import io
                reader = PdfReader(io.BytesIO(content))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except ImportError:
                raise HTTPException(status_code=400, detail="PDF support requires pypdf package")
        else:
            text = content.decode("utf-8", errors="ignore")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file")

        result = await orchestrator.review_contract_only(text, jurisdiction)
        result["filename"] = file.filename
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Full 10-Step DPA Pipeline ───
class PipelineRequest(BaseModel):
    contract_text: str
    jurisdiction: str | None = "Germany"
    assignee: str | None = "Dr. Peter"
    matter_name: str | None = "Uploaded DPA"


@app.post("/api/pipeline/run")
async def run_pipeline(req: PipelineRequest):
    """
    Full 10-step DPA agent pipeline:
    1. Upload   — contract received
    2. Intake   — classify matter & parties
    3. Clauses  — extract key DPA terms
    4. GDPR     — compliance check (parallel with 3)
    5. Risk     — score the DPA
    6. Redlines — drafting agent proposes fallback clauses
    7. QA Memo  — final legal memo
    8. Approval — Slack notification to executive
    9. Pending  — awaiting executive sign-off
    10. Closed  — audit trail finalised
    """
    from backend.integrations import slack_approval as slack

    matter_id = f"DPA-{uuid.uuid4().hex[:6].upper()}"
    audit = []

    def log_step(num, agent, action, details=""):
        entry = {"step": num, "agent": agent, "action": action,
                 "details": details, "timestamp": datetime.now().isoformat()}
        audit.append(entry)
        return entry

    def rag_context(query: str, collection: str = "legal_knowledge", n: int = 3) -> str:
        """Pull RAG context snippets — silently skips if collection missing."""
        try:
            from data_pipeline.vector_store import search_vector_store
            hits = search_vector_store(query, collection, n)
            snippets = [f"[{h.get('source','KB')}]: {(h.get('text') or '')[:300]}" for h in hits if (h.get('distance') or 9) < 1.8]
            return "\n".join(snippets) if snippets else ""
        except Exception:
            return ""

    try:
        MAX_CHARS = 5000
        text = req.contract_text[:MAX_CHARS] if len(req.contract_text) > MAX_CHARS else req.contract_text

        # ── STEP 1: Received ──────────────────────────────────────────
        log_step(1, "system", "contract_received", f"{len(req.contract_text):,} chars ingested")

        # ── STEP 2: Intake — classify matter ─────────────────────────
        log_step(2, "intake_agent", "classifying")
        intake_resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an intake agent. Classify this contract. Return JSON: {\"contract_type\":\"...\",\"parties\":[],\"jurisdiction\":\"...\",\"subject_matter\":\"...\",\"urgency\":\"HIGH|MEDIUM|LOW\"}"},
                {"role": "user", "content": f"Contract:\n{text[:1500]}"}
            ],
            response_format={"type": "json_object"}, max_tokens=200, temperature=0.1
        )
        intake = json.loads(intake_resp.choices[0].message.content)
        log_step(2, "intake_agent", "classified", f"{intake.get('contract_type','DPA')} — {intake.get('urgency','MEDIUM')} urgency")

        # ── STEPS 3 + 4: Clause extraction + GDPR compliance (PARALLEL)
        log_step(3, "clause_agent", "extracting_clauses")
        log_step(4, "compliance_agent", "gdpr_check_started")

        contract_type = intake.get('contract_type', 'Contract')
        ct_lower = contract_type.lower()

        # RAG queries adapt to contract type
        clause_rag_query = (
            f"{contract_type} manufacturing supply defect liability warranty indemnity" if any(k in ct_lower for k in ['manufactur','supply','product','procurement','vendor'])
            else f"{contract_type} litigation dispute settlement indemnity damages arbitration" if any(k in ct_lower for k in ['litigat','dispute','settl','arbitrat','claim'])
            else f"{req.matter_name} DPA clause risks GDPR" if any(k in ct_lower for k in ['dpa','data processing','gdpr'])
            else f"{contract_type} key clause risks {req.matter_name}"
        )
        compliance_rag_query = (
            f"{contract_type} product liability consumer protection warranty EU law" if any(k in ct_lower for k in ['manufactur','supply','product'])
            else f"{contract_type} litigation procedure court compliance" if any(k in ct_lower for k in ['litigat','dispute','settl'])
            else "GDPR data processing agreement Art.28 Schrems II subprocessor breach notification"
        )

        # ── All 5 RAG lookups in parallel (no GPT, just embeddings+search) ──
        loop = asyncio.get_event_loop()
        (rag_clauses, rag_gdpr, rag_policies, rag_risk, rag_memo) = await asyncio.gather(
            loop.run_in_executor(None, lambda: rag_context(clause_rag_query, "legal_knowledge", 3)),
            loop.run_in_executor(None, lambda: rag_context(compliance_rag_query, "legal_knowledge", 3)),
            loop.run_in_executor(None, lambda: rag_context(f"{req.matter_name} contract policy compliance standard", "firm_policies", 3)),
            loop.run_in_executor(None, lambda: rag_context(f"risk assessment {contract_type} {req.jurisdiction}", "legal_knowledge", 2)),
            loop.run_in_executor(None, lambda: rag_context(f"{req.matter_name} approval memo standard recommendation", "firm_policies", 3)),
        )

        clause_rag_block = f"\n\nRelevant legal knowledge (use to identify risks):\n{rag_clauses}" if rag_clauses else ""
        gdpr_rag_block = f"\n\nCompliance reference:\n{rag_gdpr}" if rag_gdpr else ""
        policy_block = f"\n\nFirm policy standards:\n{rag_policies}" if rag_policies else ""

        # Step 3 prompt adapts to contract type
        clause_system = (
            f"You are a contract clause analyst. Extract the key risk clauses from this {contract_type}. "
            "Focus on: liability caps, indemnification, warranties, defect obligations, delivery/acceptance, force majeure, termination, dispute resolution, IP ownership, penalties. "
            "Return JSON: {\"clauses\":[{\"name\":\"...\",\"text\":\"...\",\"risk\":\"HIGH|MEDIUM|LOW\",\"issue\":\"...\"}]}"
        )
        # Step 4 compliance prompt adapts to contract type
        compliance_system = (
            f"You are a compliance agent reviewing a {contract_type}. "
            + ("Check for: product liability (EU Product Liability Directive), warranty obligations, CE marking requirements, supply chain due diligence (LkSG), health & safety. "
               if any(k in ct_lower for k in ['manufactur','supply','product','procurement'])
               else "Check for: litigation risk, damages exposure, limitation periods, jurisdiction clauses, enforceability of settlement terms, costs. "
               if any(k in ct_lower for k in ['litigat','dispute','settl','arbitrat'])
               else "Check for: Schrems II transfer safeguards, subprocessor obligations, breach notification, Art.28 GDPR requirements. ")
            + "Return JSON: {\"gdpr_score\":7,\"transfer_mechanism\":\"SCCs|N/A\",\"subprocessor_clause\":true,\"breach_notification\":true,\"issues\":[{\"article\":\"...\",\"issue\":\"...\",\"severity\":\"HIGH\"}]}"
        )

        clauses_result, gdpr_result = await asyncio.gather(
            loop.run_in_executor(None, lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": clause_system},
                    {"role": "user", "content": text + clause_rag_block + policy_block}
                ],
                response_format={"type": "json_object"}, max_tokens=600, temperature=0.1
            )),
            loop.run_in_executor(None, lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": compliance_system},
                    {"role": "user", "content": text + gdpr_rag_block}
                ],
                response_format={"type": "json_object"}, max_tokens=500, temperature=0.1
            ))
        )
        clauses = json.loads(clauses_result.choices[0].message.content)
        gdpr = json.loads(gdpr_result.choices[0].message.content)
        log_step(3, "clause_agent", "clauses_extracted", f"{len(clauses.get('clauses',[]))} key clauses found | RAG-enhanced")
        log_step(4, "compliance_agent", "gdpr_check_complete", f"GDPR score: {gdpr.get('gdpr_score','N/A')}/10 — {len(gdpr.get('issues',[]))} issues | RAG-enhanced")

        # ── STEPS 5 + 6: Risk scoring AND Redlines in PARALLEL ───────
        log_step(5, "risk_agent", "scoring")
        log_step(6, "drafting_agent", "drafting_redlines")
        high_risk = [c for c in clauses.get("clauses", []) if c.get("risk") == "HIGH"]
        if not high_risk:
            high_risk = clauses.get("clauses", [])[:3]
        risk_rag_block = f"\n\nRelevant risk precedents:\n{rag_risk}" if rag_risk else ""

        risk_result, redlines_result = await asyncio.gather(
            loop.run_in_executor(None, lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Risk scoring agent. Return JSON: {\"risk_score\":0-10,\"risk_level\":\"HIGH|MEDIUM|LOW\",\"top_risks\":[\"...\"],\"safe_to_sign\":true/false,\"executive_summary\":\"2 sentences\"}"},
                    {"role": "user", "content": f"Clauses: {json.dumps(clauses)}\nGDPR: {json.dumps(gdpr)}\nContract snippet: {text[:800]}{risk_rag_block}"}
                ],
                response_format={"type": "json_object"}, max_tokens=300, temperature=0.1
            )),
            loop.run_in_executor(None, lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Contract drafting agent. For each HIGH risk clause propose a redline. Return JSON: {\"redlines\":[{\"clause\":\"...\",\"original_issue\":\"...\",\"proposed_redline\":\"...\",\"rationale\":\"...\"}]}"},
                    {"role": "user", "content": f"High risk clauses: {json.dumps(high_risk[:3])}\nCompliance issues: {json.dumps(gdpr.get('issues',[])[:3])}"}
                ],
                response_format={"type": "json_object"}, max_tokens=600, temperature=0.2
            ))
        )
        risk = json.loads(risk_result.choices[0].message.content)
        redlines = json.loads(redlines_result.choices[0].message.content)
        log_step(5, "risk_agent", "risk_scored", f"Risk: {risk.get('risk_score','?')}/10 — {risk.get('risk_level','?')} | RAG-enhanced")
        log_step(6, "drafting_agent", "redlines_ready", f"{len(redlines.get('redlines',[]))} redlines proposed")

        # ── STEP 7: QA Memo ───────────────────────────────────────────
        log_step(7, "qa_agent", "generating_memo")
        memo_rag_block = f"\n\nFirm policy standards to reference:\n{rag_memo}" if rag_memo else ""
        memo_resp = await loop.run_in_executor(None, lambda: openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a QA agent. Write a concise legal memo (3-4 paragraphs) for the executive approver. Return JSON: {\"memo\":\"...\",\"recommendation\":\"APPROVE|REJECT|NEGOTIATE\",\"conditions\":[\"...\"],\"verified_by\":\"Harveyy AI QA Agent\"}"},
                {"role": "user", "content": f"Matter: {req.matter_name}\nRisk: {json.dumps(risk)}\nGDPR issues: {json.dumps(gdpr.get('issues',[]))}\nRedlines available: {len(redlines.get('redlines',[]))}{memo_rag_block}"}
            ],
            response_format={"type": "json_object"}, max_tokens=500, temperature=0.2
        ))
        memo = json.loads(memo_resp.choices[0].message.content)

        # ── LDA clause check on top HIGH-risk clause (if configured) ──
        lda_clause_analysis = None
        try:
            lda = _get_lda()
            if lda.is_configured:
                top_clause = high_risk[0] if high_risk else (clauses.get("clauses") or [{}])[0]
                clause_txt = top_clause.get("text") or top_clause.get("name") or ""
                if clause_txt:
                    lda_result = lda.clause_check(clause_txt[:800], "Aktionsmodul Arbeitsrecht")
                    lda_clause_analysis = lda_result
                    log_step(7, "lda_agent", "clause_checked", "Otto Schmidt clause analysis complete")
        except Exception:
            pass

        log_step(7, "qa_agent", "memo_complete", f"Recommendation: {memo.get('recommendation','?')}")

        # ── STEP 8: Send technical memo to LEGAL TEAM for review
        log_step(8, "approval_agent", "notifying_legal_team")
        memo_text = memo.get("memo", "")
        legal_webhook = _get_webhook_by_role("legal_team") or os.getenv("SLACK_WEBHOOK_URL", "")
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5500")
        simplify_url = f"{frontend_url}?action=simplify&matter_id={matter_id}"
        legal_msg = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": f"📌 Legal Team Review Required — {matter_id}", "emoji": True}},
                {"type": "section", "fields": [
                    {"type": "mrkdwn", "text": f"*Matter:*\n{req.matter_name}"},
                    {"type": "mrkdwn", "text": f"*Risk Level:*\n{risk.get('risk_level','?')}"},
                    {"type": "mrkdwn", "text": f"*GDPR Score:*\n{gdpr.get('gdpr_score','?')}/10"},
                    {"type": "mrkdwn", "text": f"*Recommendation:*\n{memo.get('recommendation','?')}"},
                ]},
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*QA Memo (Technical):*\n{memo_text[:600]}..."}},
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn",
                    "text": f"✅ *Your action required:*\nOpen the Harveyy dashboard, review this memo, add your notes if needed, then click *Generate Executive Draft* to create a plain-English version and send it to Dr. Peter.\n\n👉 <{frontend_url}#matter={matter_id}|Open in Harveyy Dashboard →>"}},
                {"type": "context", "elements": [{"type": "mrkdwn", "text": f"Matter ID: `{matter_id}` | Harveyy AI | {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC"}]}
            ]
        }
        legal_sent = _send_slack_raw(legal_webhook, legal_msg)
        log_step(8, "approval_agent", "legal_team_notified", f"Legal Slack {'sent' if legal_sent else 'failed (no legal_team webhook)'} | Matter {matter_id}")

        # ── STEPS 9-10: Pending legal review + audit
        log_step(9, "system", "awaiting_legal_review", f"Matter {matter_id} pending legal team sign-off before executive")
        log_step(10, "system", "audit_trail_open", f"Pipeline complete — awaiting legal → executive approval chain")

        result = {
            "matter_id": matter_id,
            "status": "awaiting_legal_review",
            "pipeline_complete": True,
            "steps_completed": 10,
            "intake": intake,
            "clauses": clauses,
            "gdpr": gdpr,
            "risk": risk,
            "redlines": redlines,
            "memo": memo,
            "legal_slack_sent": legal_sent,
            "audit_trail": audit,
            "lda_clause_analysis": lda_clause_analysis,
        }
        _pipeline_store[matter_id] = result
        return result

    except Exception as e:
        log_step(0, "system", "pipeline_error", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ─── Legal Team → Simplify & Forward to Executive ───
@app.get("/api/pipeline/{matter_id}/status")
async def pipeline_status(matter_id: str):
    """Get stored pipeline result for a matter."""
    data = _pipeline_store.get(matter_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Matter {matter_id} not found")
    return data


class ApprovalDecisionRequest(BaseModel):
    comment: str = ""


def _exec_decision_notify(matter_id: str, decision: str, comment: str, emoji: str):
    """Store decision and notify legal team via Slack."""
    data = _pipeline_store.get(matter_id, {})
    matter_name = data.get("matter_name", matter_id)
    _pipeline_store[matter_id]["approval_status"] = decision
    _pipeline_store[matter_id]["approval_comment"] = comment
    _pipeline_store[matter_id][f"{decision.lower()}_at"] = datetime.now().isoformat()
    legal_webhook = _get_webhook_by_role("legal_team") or os.getenv("SLACK_WEBHOOK_URL", "")
    if legal_webhook:
        label = {"APPROVED": "APPROVED ✅", "REJECTED": "REJECTED ❌", "NEGOTIATE": "SENT BACK FOR CHANGES ✏️"}.get(decision, decision)
        msg = f"{emoji} *Dr. Peter has {label}*\n*Matter:* {matter_name} (`{matter_id}`)"
        if comment:
            msg += f"\n*His note:* _{comment}_"
        _send_slack_raw(legal_webhook, {"text": msg})


@app.post("/api/approvals/{matter_id}/approve")
async def approve_matter(matter_id: str, req: ApprovalDecisionRequest = ApprovalDecisionRequest()):
    """Executive approves the matter."""
    if matter_id not in _pipeline_store:
        raise HTTPException(status_code=404, detail=f"Matter {matter_id} not found")
    _exec_decision_notify(matter_id, "APPROVED", req.comment or "", "✅")
    return {"matter_id": matter_id, "decision": "APPROVED", "comment": req.comment}


@app.post("/api/approvals/{matter_id}/reject")
async def reject_matter(matter_id: str, req: ApprovalDecisionRequest = ApprovalDecisionRequest()):
    """Executive rejects the matter."""
    if matter_id not in _pipeline_store:
        raise HTTPException(status_code=404, detail=f"Matter {matter_id} not found")
    _exec_decision_notify(matter_id, "REJECTED", req.comment or "", "❌")
    return {"matter_id": matter_id, "decision": "REJECTED", "comment": req.comment}


@app.post("/api/approvals/{matter_id}/negotiate")
async def negotiate_matter(matter_id: str, req: ApprovalDecisionRequest = ApprovalDecisionRequest()):
    """Executive sends back for changes."""
    if matter_id not in _pipeline_store:
        raise HTTPException(status_code=404, detail=f"Matter {matter_id} not found")
    _exec_decision_notify(matter_id, "NEGOTIATE", req.comment or "", "✏️")
    return {"matter_id": matter_id, "decision": "NEGOTIATE", "comment": req.comment}


class SimplifyRequest(BaseModel):
    legal_notes: str = ""  # Optional corrections from legal team


@app.post("/api/pipeline/{matter_id}/simplify")
async def simplify_and_forward(matter_id: str, req: SimplifyRequest = SimplifyRequest()):
    """
    Legal team calls this after reviewing the technical memo.
    GPT rewrites into plain English for the executive approver,
    then sends to the executive Slack target.
    """
    data = _pipeline_store.get(matter_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Matter {matter_id} not found in pipeline store")

    memo_text = data.get("memo", {}).get("memo", "")
    risk = data.get("risk", {})
    gdpr = data.get("gdpr", {})
    redlines = data.get("redlines", {})
    legal_notes = req.legal_notes.strip()

    correction_block = f"\n\nLegal team corrections/notes:\n{legal_notes}" if legal_notes else ""
    original_risk_score = risk.get("risk_score", 10)
    original_risk_level = risk.get("risk_level", "HIGH")

    # ── Step A: Re-score risk AFTER applying redlines + legal notes ────────────
    redline_list = redlines.get("redlines", [])
    rescore_resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a senior legal risk analyst. A contract had an initial risk score. "
                "The legal team has now proposed redlines and corrections. "
                "Re-assess the RESIDUAL risk after these redlines are applied. "
                "A well-negotiated contract with all redlines accepted should score 0-3/10. "
                "Return JSON: {\"post_legal_risk_score\": 0-10, \"post_legal_risk_level\": \"HIGH|MEDIUM|LOW\", "
                "\"residual_risks\": [\"...\"], \"risk_reduction_summary\": \"...\"}"
            )},
            {"role": "user", "content": (
                f"Original risk score: {original_risk_score}/10 ({original_risk_level})\n"
                f"Original risk factors: {risk.get('top_risks', [])}\n\n"
                f"Proposed redlines ({len(redline_list)} total):\n"
                + "\n".join(f"- {r.get('clause','')}: {r.get('proposed_redline','')[:150]}" for r in redline_list[:5])
                + f"\n\nGDPR issues resolved: {len(data.get('gdpr',{}).get('issues',[]))}"
                + (f"\n\nLegal team notes: {legal_notes}" if legal_notes else "")
            )}
        ],
        response_format={"type": "json_object"}, max_tokens=300, temperature=0.1
    )
    rescore = json.loads(rescore_resp.choices[0].message.content)
    post_risk_score = rescore.get("post_legal_risk_score", original_risk_score)
    post_risk_level = rescore.get("post_legal_risk_level", original_risk_level)
    risk_reduction_summary = rescore.get("risk_reduction_summary", "")
    residual_risks = rescore.get("residual_risks", [])

    # ── Step B: Plain-English summary for CEO ─────────────────────────────────
    simplify_resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a senior legal officer translating a technical legal memo into plain English "
                "for a non-lawyer executive (Mr. Peter). Use simple language, no legal jargon. "
                "The legal team has already negotiated the contract and reduced the risk. "
                "Reflect the POST-negotiation risk level in your summary. "
                "Return JSON: {\"executive_summary\":\"...\",\"plain_risks\":[\"...\"],"
                "\"recommendation\":\"APPROVE|REJECT|NEGOTIATE\",\"decision_needed\":\"...\"}"
            )},
            {"role": "user", "content": (
                f"Technical memo:\n{memo_text}\n\n"
                f"ORIGINAL risk: {original_risk_score}/10 ({original_risk_level})\n"
                f"POST-NEGOTIATION risk: {post_risk_score}/10 ({post_risk_level})\n"
                f"Risk reduction: {risk_reduction_summary}\n"
                f"Remaining concerns: {residual_risks}\n"
                f"GDPR score: {gdpr.get('gdpr_score','?')}/10\n"
                f"Redlines applied: {len(redline_list)}\n"
                f"{correction_block}"
            )}
        ],
        response_format={"type": "json_object"}, max_tokens=500, temperature=0.3
    )
    simplified = json.loads(simplify_resp.choices[0].message.content)

    # Send plain-English version to executive (Mr. Peter)
    exec_webhook = _get_webhook_by_role("executive") or os.getenv("SLACK_WEBHOOK_URL", "")
    exec_summary = simplified.get("executive_summary", "")
    plain_risks = simplified.get("plain_risks", [])
    decision = simplified.get("decision_needed", "")
    recommendation = simplified.get("recommendation", "?")
    rec_emoji = {"APPROVE": "✅", "REJECT": "❌", "NEGOTIATE": "⚠️"}.get(recommendation, "❓")

    risk_delta_text = f"~{original_risk_score}/10~ → *{post_risk_score}/10* after legal review"
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5500")
    approve_url = f"{frontend_url}#approve={matter_id}"
    exec_msg = {
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": f"{rec_emoji} Contract Decision Required — {matter_id}", "emoji": True}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Hi Dr. Peter,*\n\n{exec_summary}"}},
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn",
                "text": f"*Risk after legal review:* {risk_delta_text}\n_{risk_reduction_summary}_"}},
            {"type": "section", "text": {"type": "mrkdwn",
                "text": "*Remaining concerns:*\n" + "\n".join(f"• {r}" for r in (plain_risks or residual_risks)[:3])}},
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn",
                "text": f"*Legal team recommends:* {rec_emoji} *{recommendation}*\n\n*Your decision:* {decision}"}},
            {"type": "actions", "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": "✅ Open & Approve / Reject", "emoji": True},
                 "url": approve_url, "style": "primary"},
            ]},
            {"type": "context", "elements": [{"type": "mrkdwn",
                "text": f"Reviewed by Legal Team | Matter `{matter_id}` | Harveyy AI | {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC"}]}
        ]
    }
    exec_sent = _send_slack_raw(exec_webhook, exec_msg)

    # Update stored pipeline data
    _pipeline_store[matter_id]["simplified_memo"] = simplified
    _pipeline_store[matter_id]["post_legal_risk"] = {
        "score": post_risk_score, "level": post_risk_level,
        "residual_risks": residual_risks, "reduction_summary": risk_reduction_summary,
        "original_score": original_risk_score, "original_level": original_risk_level,
    }
    _pipeline_store[matter_id]["status"] = "awaiting_executive_approval"
    _pipeline_store[matter_id]["executive_slack_sent"] = exec_sent
    if legal_notes:
        _pipeline_store[matter_id]["legal_notes"] = legal_notes

    return {
        "matter_id": matter_id,
        "status": "awaiting_executive_approval",
        "simplified": simplified,
        "post_legal_risk": _pipeline_store[matter_id]["post_legal_risk"],
        "executive_slack_sent": exec_sent,
        "message": f"Post-legal risk: {post_risk_score}/10. Plain-English summary sent to Dr. Peter {'successfully' if exec_sent else '(no executive webhook configured)'}"
    }


# ─── Templates ───
@app.get("/api/templates")
async def list_templates():
    """List available contract templates."""
    return orchestrator.get_templates()


# ─── LDA Legal Data Hub (Otto Schmidt) ───────────────────────────────────────
def _get_lda():
    from backend.integrations.lda_client import LDAClient
    return LDAClient()


@app.get("/api/lda/status")
async def lda_status():
    """Check if LDA is configured and list available data assets."""
    lda = _get_lda()
    if not lda.is_configured:
        return {"configured": False, "message": "Set LDA_CLIENT_ID and LDA_CLIENT_SECRET in .env"}
    try:
        assets = lda.list_data_assets()
        return {"configured": True, "data_assets": assets}
    except Exception as e:
        return {"configured": True, "error": str(e), "data_assets": []}


class LDASearchRequest(BaseModel):
    query: str
    data_asset: str = "Beratermodul Miet- und WEG-Recht"
    size: int = 5


class LDAQnARequest(BaseModel):
    question: str
    data_asset: str = "Beratermodul Miet- und WEG-Recht"
    mode: str = "attribution"


class LDAClauseCheckRequest(BaseModel):
    clause_text: str
    data_asset: str = "Aktionsmodul Arbeitsrecht"


@app.post("/api/lda/search")
async def lda_search(req: LDASearchRequest):
    """Keyword search across LDA Otto Schmidt data assets."""
    try:
        lda = _get_lda()
        if not lda.is_configured:
            raise HTTPException(status_code=400, detail="LDA not configured — add LDA_CLIENT_ID and LDA_CLIENT_SECRET to .env")
        result = lda.search(req.query, req.data_asset, req.size)
        hits = result.get("hits", {}).get("hits", [])
        return {
            "query": req.query,
            "data_asset": req.data_asset,
            "total": result.get("hits", {}).get("total", {}).get("value", 0),
            "results": [
                {
                    "id": h.get("_id"),
                    "score": h.get("_score"),
                    "source": h.get("_source", {}).get("metadata", {}).get("dokumententyp", ""),
                    "date": h.get("_source", {}).get("metadata", {}).get("datum", ""),
                    "reference": h.get("_source", {}).get("metadata", {}).get("aktenzeichen", ""),
                    "url": h.get("_source", {}).get("metadata", {}).get("oso_url", ""),
                    "leitsatz": h.get("_source", {}).get("metadata", {}).get("leitsatz", ""),
                    "text": (h.get("_source", {}).get("text") or "")[:500],
                    "highlight": " … ".join(h.get("highlight", {}).get("text", [])),
                }
                for h in hits
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lda/semantic-search")
async def lda_semantic_search(req: LDASearchRequest):
    """AI semantic search across LDA Otto Schmidt data assets."""
    try:
        lda = _get_lda()
        if not lda.is_configured:
            raise HTTPException(status_code=400, detail="LDA not configured")
        result = lda.semantic_search(req.query, req.data_asset, req.size)
        return {"query": req.query, "data_asset": req.data_asset, "results": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lda/qna")
async def lda_qna(req: LDAQnARequest):
    """Ask a legal question — gets AI answer with Otto Schmidt source attribution."""
    try:
        lda = _get_lda()
        if not lda.is_configured:
            raise HTTPException(status_code=400, detail="LDA not configured — add LDA_CLIENT_ID and LDA_CLIENT_SECRET to .env")
        result = lda.qna(req.question, req.data_asset, req.mode)
        return {"question": req.question, "data_asset": req.data_asset, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lda/clause-check")
async def lda_clause_check(req: LDAClauseCheckRequest):
    """Check a contract clause against Otto Schmidt legal database."""
    try:
        lda = _get_lda()
        if not lda.is_configured:
            raise HTTPException(status_code=400, detail="LDA not configured")
        result = lda.clause_check(req.clause_text, req.data_asset)
        return {"clause": req.clause_text[:200], "data_asset": req.data_asset, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── RAG Search (direct vector store query) ───
class RAGSearchRequest(BaseModel):
    query: str
    collection: str = "legal_knowledge"
    n_results: int = 5


@app.post("/api/rag/search")
async def rag_search(req: RAGSearchRequest):
    """Search the legal knowledge base directly (RAG)."""
    try:
        from data_pipeline.vector_store import search_vector_store
        results = search_vector_store(req.query, req.collection, req.n_results)
        return {"query": req.query, "results": results, "collection": req.collection}
    except Exception as e:
        return {"query": req.query, "results": [], "error": str(e)}


# ─── Compare Contracts ───
class CompareRequest(BaseModel):
    contract_a: str
    contract_b: str


@app.post("/api/compare")
async def compare_contracts(req: CompareRequest):
    """Compare two contract versions."""
    try:
        result = await orchestrator.contract_review.compare_contracts(
            req.contract_a, req.contract_b
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Knowledge Base: Upload Document ───
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".json", ".csv", ".docx"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


def _validate_upload(file: UploadFile):
    """Validate file before processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    return ext


async def _extract_text(file: UploadFile, content: bytes, ext: str) -> str:
    """Safely extract text from uploaded file content."""
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large: {len(content)/(1024*1024):.1f}MB (max 50MB)")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            text = ""
        # Fallback: OCR via pymupdf + OpenAI Vision for scanned PDFs
        if len(text.strip()) < 200:
            try:
                import fitz, base64
                doc = fitz.open(stream=content, filetype="pdf")
                ocr_texts = []
                for i in range(min(doc.page_count, 6)):
                    pix = doc[i].get_pixmap(dpi=100)
                    b64 = base64.b64encode(pix.tobytes("png")).decode()
                    resp = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
                            {"type": "text", "text": "Extract all text from this document page. Return only the text."}
                        ]}], max_tokens=1500,
                    )
                    ocr_texts.append(resp.choices[0].message.content.strip())
                text = "\n\n".join(ocr_texts)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"PDF OCR failed: {e}")
    else:
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = content.decode("latin-1", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from file")
    return text


@app.post("/api/kb/upload")
async def kb_upload_document(
    file: UploadFile = File(...),
    collection: str = Form("firm_policies"),
):
    """Upload a document and index it into the knowledge base."""
    ext = _validate_upload(file)
    try:
        content = await file.read()
        text = await _extract_text(file, content, ext)

        from data_pipeline.vector_store import ingest_document
        n_chunks = ingest_document(text, file.filename, collection)
        return {
            "status": "indexed",
            "filename": file.filename,
            "collection": collection,
            "chunks_created": n_chunks,
            "text_length": len(text),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ─── Knowledge Base: Bulk Upload (multiple files) ───
@app.post("/api/kb/upload-multiple")
async def kb_upload_multiple(
    files: list[UploadFile] = File(...),
    collection: str = Form("firm_policies"),
):
    """Upload multiple documents and index them. Never crashes — reports per-file status."""
    from data_pipeline.vector_store import ingest_document
    results = []
    succeeded = 0
    for file in files:
        fname = file.filename or "unnamed"
        try:
            ext = "." + fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            if ext not in ALLOWED_EXTENSIONS:
                results.append({"filename": fname, "status": "skipped", "error": f"Unsupported type: {ext}"})
                continue

            content = await file.read()
            if len(content) == 0:
                results.append({"filename": fname, "status": "skipped", "error": "Empty file"})
                continue
            if len(content) > MAX_UPLOAD_SIZE:
                results.append({"filename": fname, "status": "skipped", "error": "File too large (>50MB)"})
                continue

            if ext == ".pdf":
                try:
                    from pypdf import PdfReader
                    import io
                    reader = PdfReader(io.BytesIO(content))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception as e:
                    results.append({"filename": fname, "status": "error", "error": f"PDF read failed: {e}"})
                    continue
            else:
                try:
                    text = content.decode("utf-8", errors="ignore")
                except Exception:
                    text = content.decode("latin-1", errors="ignore")

            if not text.strip():
                results.append({"filename": fname, "status": "skipped", "error": "No text extracted"})
                continue

            n = ingest_document(text, fname, collection)
            results.append({"filename": fname, "chunks": n, "status": "ok"})
            succeeded += 1
        except Exception as e:
            results.append({"filename": fname, "status": "error", "error": str(e)})

    try:
        from data_pipeline.vector_store import list_collections
        cols = list_collections()
        total_chunks = next((c["count"] for c in cols if c["name"] == collection), 0)
    except Exception:
        total_chunks = None
    return {"collection": collection, "files": results, "total_files": len(results), "succeeded": succeeded, "collection_total_chunks": total_chunks}


# ─── Knowledge Base: Import Folder ───
class FolderImportRequest(BaseModel):
    folder_path: str
    collection: str = "firm_policies"


@app.post("/api/kb/import-folder")
async def kb_import_folder(req: FolderImportRequest):
    """Import all documents from a folder into the knowledge base."""
    import os
    folder = req.folder_path.strip()
    if not folder:
        raise HTTPException(status_code=400, detail="Folder path cannot be empty")
    if not os.path.exists(folder):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Not a directory: {folder}")
    try:
        from data_pipeline.vector_store import ingest_folder
        result = ingest_folder(folder, req.collection)
        result["collection"] = req.collection
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Folder import failed: {str(e)}")


# ─── Knowledge Base: List Collections ───
@app.get("/api/kb/collections")
async def kb_list_collections():
    """List all knowledge base collections and their sizes."""
    try:
        from data_pipeline.vector_store import list_collections
        return {"collections": list_collections()}
    except Exception as e:
        return {"collections": [], "error": str(e)}


# ─── LDA Legal Data Hub Integration ───
class LDASearchRequest(BaseModel):
    query: str
    data_asset: str = "Beratermodul Miet- und WEG-Recht"
    size: int = 10


class LDASemanticRequest(BaseModel):
    query: str
    data_asset: str = "Aktionsmodul Familienrecht"
    candidates: int = 5


class LDAQnARequest(BaseModel):
    question: str
    data_asset: str = "Beratermodul Miet- und WEG-Recht"
    mode: str = "attribution"


class LDAClauseCheckRequest(BaseModel):
    clause_text: str
    data_asset: str = "Aktionsmodul Arbeitsrecht"
    mode: str = "check"


def _get_lda_client():
    from backend.integrations.lda_client import LDAClient
    return LDAClient()


@app.get("/api/lda/status")
async def lda_status():
    """Check if LDA is configured."""
    try:
        client = _get_lda_client()
        return {"configured": client.is_configured, "api_base": client.api_base}
    except Exception as e:
        return {"configured": False, "error": str(e)}


@app.get("/api/lda/data-assets")
async def lda_data_assets():
    """List available LDA data assets."""
    try:
        client = _get_lda_client()
        if not client.is_configured:
            return {"error": "LDA not configured. Set LDA_CLIENT_ID and LDA_CLIENT_SECRET in .env"}
        return client.list_data_assets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lda/search")
async def lda_search(req: LDASearchRequest):
    """Keyword search via LDA Legal Data Hub."""
    try:
        client = _get_lda_client()
        if not client.is_configured:
            return {"error": "LDA not configured. Set LDA_CLIENT_ID and LDA_CLIENT_SECRET in .env"}
        return client.search(req.query, req.data_asset, req.size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lda/semantic-search")
async def lda_semantic_search(req: LDASemanticRequest):
    """Semantic search via LDA Legal Data Hub."""
    try:
        client = _get_lda_client()
        if not client.is_configured:
            return {"error": "LDA not configured. Set LDA_CLIENT_ID and LDA_CLIENT_SECRET in .env"}
        return client.semantic_search(req.query, req.data_asset, req.candidates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lda/qna")
async def lda_qna(req: LDAQnARequest):
    """Ask a legal question via LDA QnA."""
    try:
        client = _get_lda_client()
        if not client.is_configured:
            return {"error": "LDA not configured. Set LDA_CLIENT_ID and LDA_CLIENT_SECRET in .env"}
        return client.qna(req.question, req.data_asset, req.mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lda/clause-check")
async def lda_clause_check(req: LDAClauseCheckRequest):
    """Check a clause via LDA Clause Check."""
    try:
        client = _get_lda_client()
        if not client.is_configured:
            return {"error": "LDA not configured. Set LDA_CLIENT_ID and LDA_CLIENT_SECRET in .env"}
        return client.clause_check(req.clause_text, req.data_asset, req.mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Partner Approval Workflow ───
class ApprovalActionRequest(BaseModel):
    note: str = ""


@app.get("/api/approvals/status")
async def approvals_status():
    """Check if Slack approval is configured."""
    from backend.integrations.slack_approval import is_configured
    return {"configured": is_configured(), "webhook_set": bool(os.getenv("SLACK_WEBHOOK_URL", ""))}


@app.get("/api/approvals")
async def list_approvals():
    """List all pending and past approvals."""
    from backend.integrations.slack_approval import list_pending
    return {"approvals": list_pending()}


@app.get("/api/approvals/{approval_id}")
async def get_approval(approval_id: str):
    """Get a specific approval."""
    from backend.integrations.slack_approval import get_approval as _get
    result = _get(approval_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Approval {approval_id} not found")
    return result


@app.post("/api/approvals/{approval_id}/approve")
async def approve_request(approval_id: str, req: ApprovalActionRequest = ApprovalActionRequest()):
    """Partner approves a draft — clears it for sending to client."""
    from backend.integrations.slack_approval import approve
    result = approve(approval_id, req.note)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/approvals/{approval_id}/reject")
async def reject_request(approval_id: str, req: ApprovalActionRequest = ApprovalActionRequest()):
    """Partner rejects a draft — blocks sending, with optional note."""
    from backend.integrations.slack_approval import reject
    result = reject(approval_id, req.note)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/draft-with-approval")
async def draft_with_approval(req: DraftRequest):
    """Draft a contract AND send it for partner approval before delivery."""
    try:
        result = await orchestrator.draft_contract(req.requirements)

        # Send to partner for approval via Slack
        from backend.integrations.slack_approval import send_approval_request
        contract_text = ""
        if result.get("agent_results", {}).get("drafting"):
            contract_text = result["agent_results"]["drafting"].get("contract_text", "")
        elif result.get("agent_results", {}).get("qa_review"):
            contract_text = str(result["agent_results"]["qa_review"])

        approval = send_approval_request(
            matter_id=result.get("id", "unknown"),
            doc_type="contract_draft",
            content=contract_text,
            summary=f"Contract draft: {req.requirements[:100]}",
            recipient="client",
        )

        result["approval"] = approval
        result["status"] = "pending_approval"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Agent Status ───
@app.get("/api/agents")
async def agent_status():
    """Get status of all agents."""
    return {
        "agents": [
            {
                "id": "intake",
                "name": "Intake Agent",
                "role": "Reception / Paralegal",
                "status": "active",
                "description": "Classifies legal requests and extracts key entities",
            },
            {
                "id": "contract_review",
                "name": "Contract Review Agent",
                "role": "Associate Attorney",
                "status": "active",
                "description": "Analyzes contracts clause-by-clause, flags risks",
            },
            {
                "id": "legal_research",
                "name": "Legal Research Agent",
                "role": "Research Librarian",
                "status": "active",
                "description": "RAG-powered legal research with German + US law",
            },
            {
                "id": "drafting",
                "name": "Drafting Agent",
                "role": "Senior Associate",
                "status": "active",
                "description": "Generates contracts from templates and requirements",
            },
            {
                "id": "orchestrator",
                "name": "Orchestrator",
                "role": "Managing Partner",
                "status": "active",
                "description": "Routes tasks, manages workflow, ensures quality",
            },
        ]
    }
