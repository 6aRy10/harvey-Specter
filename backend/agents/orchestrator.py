"""
Orchestrator Agent — The Managing Partner
==========================================
Routes tasks between agents, manages workflow, ensures quality,
and provides a unified interface for the client.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from openai import OpenAI

from .contract_review_agent import ContractReviewAgent
from .drafting_agent import DraftingAgent
from .intake_agent import IntakeAgent
from .legal_research_agent import LegalResearchAgent


class Matter:
    """Represents a legal matter / case being handled."""

    def __init__(self, matter_id: str, client_request: str):
        self.id = matter_id
        self.client_request = client_request
        self.created_at = datetime.now().isoformat()
        self.status = "intake"
        self.classification = None
        self.entities = None
        self.agent_results = {}
        self.audit_trail = []

    def add_audit_entry(self, agent: str, action: str, details: str = ""):
        self.audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "details": details,
        })

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "client_request": self.client_request,
            "created_at": self.created_at,
            "status": self.status,
            "classification": self.classification,
            "entities": self.entities,
            "agent_results": self.agent_results,
            "audit_trail": self.audit_trail,
        }


class Orchestrator:
    """
    Central orchestrator that manages the AI law firm workflow.
    
    Workflow:
    1. Client request → Intake Agent (classify & extract)
    2. Based on classification → route to specialist agent(s)
    3. Specialist agent(s) process the request
    4. Orchestrator consolidates results
    5. Quality review (optional second pass)
    6. Return results to client
    """

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.intake = IntakeAgent(openai_client)
        self.contract_review = ContractReviewAgent(openai_client)
        self.legal_research = LegalResearchAgent(openai_client)
        self.drafting = DraftingAgent(openai_client)
        self.matters: dict[str, Matter] = {}

    async def create_matter(self, client_request: str, contract_text: str = None) -> dict:
        """
        Process a new legal matter end-to-end.
        
        Args:
            client_request: The client's description of what they need
            contract_text: Optional contract text for review
            
        Returns:
            Complete matter with all agent results
        """
        matter_id = f"LEX-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        matter = Matter(matter_id, client_request)
        self.matters[matter_id] = matter

        # ─── Step 1: Intake Classification ───
        matter.add_audit_entry("orchestrator", "matter_created", f"New matter: {matter_id}")
        matter.add_audit_entry("intake", "classification_started")

        intake_result = await self.intake.process(client_request)
        matter.classification = intake_result.get("classification", [])
        matter.entities = intake_result.get("entities", {})
        matter.agent_results["intake"] = intake_result
        matter.add_audit_entry("intake", "classification_complete",
                              f"Classified as: {', '.join(matter.classification)}")

        # ─── Step 2: Route to Specialist Agents ───
        classifications = matter.classification
        context = matter.entities

        # Route based on classification
        if "CONTRACT_REVIEW" in classifications and contract_text:
            matter.status = "reviewing"
            matter.add_audit_entry("contract_review", "review_started")
            review_result = await self.contract_review.review_contract(contract_text, context)
            matter.agent_results["contract_review"] = review_result
            matter.add_audit_entry("contract_review", "review_complete",
                                  f"Risk score: {review_result.get('overall_risk_score', 'N/A')}")

        if "CONTRACT_DRAFTING" in classifications:
            matter.status = "drafting"
            matter.add_audit_entry("drafting", "drafting_started")
            draft_result = await self.drafting.draft_contract(client_request, context)
            matter.agent_results["drafting"] = draft_result
            matter.add_audit_entry("drafting", "drafting_complete",
                                  f"Template: {draft_result.get('template_used', 'custom')}")

            # Auto-review the drafted contract
            drafted_text = draft_result.get("contract_text", "")
            if drafted_text:
                matter.add_audit_entry("contract_review", "qa_review_started",
                                      "Auto-reviewing drafted contract for quality")
                qa_result = await self.contract_review.review_contract(drafted_text, context)
                matter.agent_results["qa_review"] = qa_result
                matter.add_audit_entry("contract_review", "qa_review_complete",
                                      f"QA Risk score: {qa_result.get('overall_risk_score', 'N/A')}")

        if "LEGAL_RESEARCH" in classifications:
            matter.status = "researching"
            matter.add_audit_entry("legal_research", "research_started")
            research_result = await self.legal_research.research(client_request, context)
            matter.agent_results["legal_research"] = research_result
            matter.add_audit_entry("legal_research", "research_complete",
                                  f"Confidence: {research_result.get('confidence_level', 'N/A')}")

        if "COMPLIANCE_CHECK" in classifications:
            matter.status = "researching"
            compliance_question = f"Compliance check: {client_request}"
            matter.add_audit_entry("legal_research", "compliance_check_started")
            compliance_result = await self.legal_research.research(compliance_question, context)
            matter.agent_results["compliance_check"] = compliance_result
            matter.add_audit_entry("legal_research", "compliance_check_complete")

        if "GENERAL_COUNSEL" in classifications:
            matter.status = "researching"
            matter.add_audit_entry("legal_research", "general_counsel_started")
            counsel_result = await self.legal_research.research(client_request, context)
            matter.agent_results["general_counsel"] = counsel_result
            matter.add_audit_entry("legal_research", "general_counsel_complete")

        # ─── Step 3: Consolidate Results ───
        matter.status = "complete"
        matter.add_audit_entry("orchestrator", "matter_complete",
                              f"Processed by {len(matter.agent_results)} agents")

        return matter.to_dict()

    async def review_contract_only(self, contract_text: str, jurisdiction: str = None, context: dict = None) -> dict:
        """Quick contract review without full intake."""
        ctx = context or {}
        if jurisdiction:
            ctx["jurisdiction"] = jurisdiction
        result = await self.contract_review.review_contract(contract_text, ctx if ctx else None)
        return result

    async def draft_contract_only(self, requirements: str, jurisdiction: str = None) -> dict:
        """Quick contract drafting without full intake."""
        context = {"jurisdiction": jurisdiction} if jurisdiction else None
        result = await self.drafting.draft_contract(requirements, context)
        return result

    async def research_only(self, question: str, jurisdiction: str = None) -> dict:
        """Quick legal research without full intake."""
        context = {"jurisdiction": jurisdiction} if jurisdiction else None
        result = await self.legal_research.research(question, context)
        return result

    def get_matter(self, matter_id: str) -> Optional[dict]:
        """Retrieve a matter by ID."""
        matter = self.matters.get(matter_id)
        return matter.to_dict() if matter else None

    def list_matters(self) -> list[dict]:
        """List all matters."""
        return [m.to_dict() for m in self.matters.values()]

    def get_templates(self) -> list[dict]:
        """List available contract templates."""
        return self.drafting.list_templates()
