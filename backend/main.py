"""
LexAgents API — FastAPI Backend
================================
Main API server for the AI Law Firm.

Run with: uvicorn backend.main:app --reload --port 8000
"""

import asyncio
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

        # Step 1: Contract Review
        audit.append({"agent": "contract_review", "action": "review_started", "timestamp": datetime.now().isoformat()})
        if intake_context:
            audit.append({"agent": "intake", "action": "context_provided", "details": intake_context[:200]})
        review = await orchestrator.review_contract_only(req.contract_text, req.jurisdiction, review_context if review_context else None)
        audit.append({"agent": "contract_review", "action": "review_complete", "details": f"Risk: {review.get('overall_risk_score', 'N/A')}"})

        # Step 2: Auto-research the top flagged clauses (PARALLEL for speed)
        flagged = review.get("clauses_found", [])
        research_results = []
        if flagged:
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

        return {
            "id": matter_id,
            "status": "complete",
            "agent": "contract_review",
            "agent_results": {"contract_review": review},
            "auto_research": research_results,
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
            analysis = research.get("legal_analysis", "")
            sources = research.get("sources_cited", [])
            laws = research.get("applicable_laws", [])
            lang = research.get("detected_language", "en")

            # Build cited response
            reply_parts = [analysis[:800]] if analysis else ["I couldn't find specific information on that topic."]
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


# ─── Templates ───
@app.get("/api/templates")
async def list_templates():
    """List available contract templates."""
    return orchestrator.get_templates()


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
        except ImportError:
            raise HTTPException(status_code=400, detail="PDF support requires pypdf. Run: pip install pypdf")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")
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

    return {"collection": collection, "files": results, "total_files": len(results), "succeeded": succeeded}


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
