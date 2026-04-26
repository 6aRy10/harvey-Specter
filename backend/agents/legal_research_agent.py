"""
Legal Research Agent — The Research Librarian
===============================================
RAG-powered legal research agent. Searches the German law
vector store and provides cited legal analysis.
"""

import json
import os
from pathlib import Path

import chromadb
from openai import OpenAI

RESEARCH_SYSTEM_PROMPT = """You are a Senior Legal Researcher at LexAgents, an AI law firm.
You ONLY answer questions about law, contracts, compliance, and legal matters.

## SCOPE — VERY IMPORTANT:
- You ONLY handle: contracts, clauses, GDPR/DSGVO, BGB, HGB, EU law, employment law, IP law, data protection, liability, compliance, legal risk.
- If the question is NOT about law or legal matters (e.g. cooking, sports, weather, random topics, jokes, coding, science unrelated to law), you MUST refuse with this exact JSON:
  {"out_of_scope": true, "legal_analysis": "I can only answer legal questions. This topic is outside my scope.", "key_findings": [], "recommendations": [], "sources_cited": [], "confidence_level": "NONE", "applicable_laws": [], "jurisdiction": "N/A", "detected_language": "en", "question_analyzed": "<repeat question>"}

## CRITICAL: Multilingual Support
- DETECT the language of the user's question.
- RESPOND in the SAME language the user wrote in.
- Legal terms (e.g., "BGB §622", "DSGVO Art. 28") always stay in their original form.

## Knowledge Base Usage:
- If Retrieved Legal Sources are provided and relevant, cite them directly.
- If NO relevant sources are retrieved, answer from your general legal knowledge but add a note: "Note: This answer is based on general legal knowledge — not found in the firm knowledge base."
- NEVER invent case citations or statute text you are not certain about.

## CITATION REQUIREMENTS:
- Every legal claim MUST cite a specific law and section: "BGB §622 Abs. 2" or "GDPR Art. 28(3)"
- Cite retrieved sources as "[Source: filename]"

## Output Format:
Return valid JSON:
{
    "question_analyzed": "Restated legal question",
    "detected_language": "en / de / hi",
    "jurisdiction": "Primary applicable jurisdiction",
    "out_of_scope": false,
    "kb_hit": true,
    "applicable_laws": [{"law": "...", "section": "...", "relevance": "...", "exact_citation": "..."}],
    "legal_analysis": "Detailed analysis with inline citations",
    "key_findings": ["Finding (with citation)"],
    "risks_and_considerations": ["Risk 1"],
    "recommendations": ["Action 1"],
    "sources_cited": ["Source 1"],
    "confidence_level": "HIGH / MEDIUM / LOW",
    "disclaimer": "AI-generated legal research — review with a qualified attorney."
}
"""


class LegalResearchAgent:
    def __init__(self, openai_client: OpenAI, chroma_dir: str = "data/vector_store"):
        self.client = openai_client
        self.model = "gpt-4o-mini"
        self.chroma_dir = chroma_dir
        self._chroma_client = None

    @property
    def chroma_client(self):
        if self._chroma_client is None:
            if Path(self.chroma_dir).exists():
                self._chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
            else:
                self._chroma_client = None
        return self._chroma_client

    def _search_knowledge_base(self, query: str, n_results: int = 5) -> list[dict]:
        """Search the legal knowledge vector store."""
        if not self.chroma_client:
            return []

        try:
            collection = self.chroma_client.get_collection("legal_knowledge")

            # Embed the query
            embedding_response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=[query],
            )
            query_embedding = embedding_response.data[0].embedding

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            formatted = []
            for i in range(len(results["ids"][0])):
                formatted.append({
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i].get("source", "Unknown"),
                    "relevance": 1 - results["distances"][0][i],  # Convert distance to similarity
                })
            return formatted

        except Exception:
            return []

    # Keywords that signal a legal question — fast pre-check before hitting GPT
    LEGAL_KEYWORDS = {
        "contract", "clause", "agreement", "law", "legal", "liability", "gdpr", "dsgvo", "bgb",
        "hgb", "compliance", "regulation", "breach", "termination", "jurisdiction", "damages",
        "indemnity", "warranty", "intellectual", "property", "copyright", "patent", "privacy",
        "data", "subprocessor", "processor", "controller", "dpa", "nda", "nda", "schrems",
        "employment", "arbitration", "court", "parties", "obligation", "rights", "penalty",
        "governing", "force majeure", "confidential", "eu", "european", "dsgvo", "art.", "§",
        "non-compete", "noncompete", "transfer", "processing", "retention", "notice", "ip",
    }

    def _is_legal_question(self, question: str) -> bool:
        """Fast keyword check — True if any legal term appears."""
        q = question.lower()
        return any(kw in q for kw in self.LEGAL_KEYWORDS)

    async def research(self, question: str, context: dict = None) -> dict:
        """
        Perform legal research on a question.
        - Non-legal questions → immediate out-of-scope refusal.
        - Legal questions → RAG search first; answer from KB if hits, general law if not.
        """
        # ── Pre-check: is this even a legal question? ──────────────────────────
        if not self._is_legal_question(question):
            return {
                "out_of_scope": True,
                "question_analyzed": question,
                "detected_language": "en",
                "jurisdiction": "N/A",
                "legal_analysis": "I can only answer legal questions. This topic is outside my scope as a legal research tool.",
                "key_findings": [],
                "risks_and_considerations": [],
                "recommendations": [],
                "sources_cited": [],
                "applicable_laws": [],
                "confidence_level": "NONE",
                "disclaimer": "",
                "agent": "legal_research",
                "status": "out_of_scope",
                "rag_sources_used": 0,
                "kb_hit": False,
            }

        # ── Step 1: Search the knowledge base ─────────────────────────────────
        rag_results = self._search_knowledge_base(question)
        RELEVANCE_THRESHOLD = 0.2  # relevance = 1 - distance; > 0.2 means distance < 0.8
        good_hits = [r for r in rag_results if r["relevance"] >= RELEVANCE_THRESHOLD]
        kb_hit = len(good_hits) > 0

        # ── Step 2: Build the user message ────────────────────────────────────
        user_message = f"Legal Research Question:\n{question}\n"

        if context:
            if context.get("jurisdiction"):
                user_message += f"\nJurisdiction focus: {context['jurisdiction']}"
            if context.get("industry"):
                user_message += f"\nIndustry: {context['industry']}"

        if good_hits:
            user_message += "\n\n## Retrieved Legal Sources (from firm knowledge base):\n"
            for i, result in enumerate(good_hits, 1):
                user_message += f"\n### Source {i} (relevance: {result['relevance']:.2f}, from: {result['source']})\n"
                user_message += f"{result['text']}\n"
        else:
            user_message += "\n\n[No relevant documents found in the knowledge base for this query. Answer from your general legal knowledge only, and note this in your response.]"

        # ── Step 3: Generate research analysis ────────────────────────────────
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=2000,
        )

        result = json.loads(response.choices[0].message.content)
        result["agent"] = "legal_research"
        result["status"] = "researched"
        result["rag_sources_used"] = len(good_hits)
        result["kb_hit"] = kb_hit
        result["out_of_scope"] = result.get("out_of_scope", False)
        return result
