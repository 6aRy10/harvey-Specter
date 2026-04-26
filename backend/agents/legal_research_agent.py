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
You provide thorough, well-cited legal research and analysis.

## CRITICAL: Multilingual Support
- DETECT the language of the user's question.
- RESPOND in the SAME language the user wrote in.
- If the user writes in German, respond fully in German (including analysis, findings, recommendations).
- If the user writes in Hindi, respond fully in Hindi.
- If the user writes in English, respond in English.
- Legal terms (e.g., "BGB §622", "DSGVO Art. 28") should always remain in their original form regardless of response language.

## Your Capabilities:
- German law (BGB, HGB, GDPR/DSGVO, ArbG, GmbHG, etc.)
- EU regulations and directives
- US contract law (UCC, common law)
- International commercial law

## Your Research Process:
1. Analyze the legal question
2. Search relevant statutes, regulations, and case law
3. Provide a structured legal analysis with PRECISE citations
4. For every claim, cite the exact statute section (e.g., "BGB §622 Abs. 2", "DSGVO Art. 28 Abs. 3")
5. Highlight any jurisdictional conflicts
6. Suggest practical next steps

## CITATION REQUIREMENTS (CRITICAL):
- Every legal claim MUST cite a specific law, section, and paragraph
- Use the format: "Gemäß BGB §622 Abs. 2 Nr. 1..." or "According to GDPR Art. 28(3)..."
- If you reference a retrieved source, cite it as "[Source: filename, relevance: X%]"
- Never make unsupported legal claims

## Context Documents:
You will receive relevant legal text chunks retrieved from your knowledge base.
Always cite these sources when referencing them.

## Output Format:
Return valid JSON:
{
    "question_analyzed": "Restated legal question (in detected language)",
    "detected_language": "en / de / hi",
    "jurisdiction": "Primary applicable jurisdiction",
    "applicable_laws": [
        {
            "law": "Name of statute/regulation",
            "section": "Relevant section with paragraph (e.g., §622 Abs. 2)",
            "relevance": "How it applies",
            "exact_citation": "Full citation text"
        }
    ],
    "legal_analysis": "Detailed analysis in the detected language with inline citations",
    "key_findings": ["Finding 1 (with citation)", "Finding 2 (with citation)"],
    "risks_and_considerations": ["Risk 1", "Risk 2"],
    "recommendations": ["Action 1", "Action 2"],
    "sources_cited": ["Source 1 with section reference", "Source 2"],
    "confidence_level": "HIGH / MEDIUM / LOW",
    "disclaimer": "This is AI-generated legal research and should be reviewed by a qualified attorney. / Dies ist eine KI-generierte Rechtsrecherche und sollte von einem qualifizierten Anwalt überprüft werden."
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

    async def research(self, question: str, context: dict = None) -> dict:
        """
        Perform legal research on a question.
        Uses RAG to search the knowledge base first.
        """
        # Step 1: Search the knowledge base
        rag_results = self._search_knowledge_base(question)

        # Step 2: Build the user message with retrieved context
        user_message = f"Legal Research Question:\n{question}\n"

        if context:
            if context.get("jurisdiction"):
                user_message += f"\nJurisdiction focus: {context['jurisdiction']}"
            if context.get("industry"):
                user_message += f"\nIndustry: {context['industry']}"

        if rag_results:
            user_message += "\n\n## Retrieved Legal Sources:\n"
            for i, result in enumerate(rag_results, 1):
                user_message += f"\n### Source {i} (relevance: {result['relevance']:.2f}, from: {result['source']})\n"
                user_message += f"{result['text']}\n"

        # Step 3: Generate research analysis
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=4000,
        )

        result = json.loads(response.choices[0].message.content)
        result["agent"] = "legal_research"
        result["status"] = "researched"
        result["rag_sources_used"] = len(rag_results)
        return result
