"""
Contract Review Agent — The Associate Attorney
================================================
Analyzes contracts clause by clause, flags risks,
and suggests modifications. Uses CUAD knowledge base.
"""

import json
from pathlib import Path

from openai import OpenAI

DEFAULT_REVIEW_PROMPT = """You are a Contract Review Attorney AI. Analyze the contract quickly and return JSON only.

Rules:
- Flag TOP 3 riskiest clauses only (HIGH/MEDIUM risk)
- Keep explanations under 2 sentences
- For original_text: copy a short verbatim snippet (max 80 chars) for highlighting
- List up to 2 missing protections
- Give 3 concise recommendations
- Be FAST — no verbose output

Return ONLY this JSON (no markdown, no extra text):
{
    "overall_risk_score": 7,
    "risk_level": "HIGH",
    "executive_summary": "2-3 sentence summary",
    "contract_type": "NDA",
    "parties_identified": ["Party A", "Party B"],
    "clauses_found": [
        {"clause_type": "Non-Compete", "risk_level": "HIGH", "original_text": "short verbatim snippet", "explanation": "Why risky.", "suggestion": "How to fix it.", "section": "§5"}
    ],
    "missing_protections": [
        {"clause_type": "Liability Cap", "importance": "HIGH", "recommendation": "Add a liability cap."}
    ],
    "top_recommendations": ["Rec 1", "Rec 2", "Rec 3"],
    "gdpr_compliance_notes": "One sentence on GDPR.",
    "errors_found": []
}
"""


class ContractReviewAgent:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.model = "gpt-4o-mini"
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the CUAD-enriched system prompt if available."""
        prompt_file = Path("data/processed/contract_review_prompt.txt")
        if prompt_file.exists():
            base_prompt = prompt_file.read_text(encoding="utf-8")
            return base_prompt + "\n\nAlways respond in valid JSON format."
        return DEFAULT_REVIEW_PROMPT

    async def review_contract(self, contract_text: str, context: dict = None) -> dict:
        """
        Review a contract and return structured risk analysis.
        
        Args:
            contract_text: The full text of the contract to review
            context: Optional context from intake (jurisdiction, contract type, etc.)
        """
        user_message = f"Please review the following contract:\n\n{contract_text}"

        if context:
            user_message += f"\n\nAdditional context:\n"
            if context.get("jurisdiction"):
                user_message += f"- Jurisdiction: {context['jurisdiction']}\n"
            if context.get("contract_type"):
                user_message += f"- Expected contract type: {context['contract_type']}\n"
            if context.get("key_issues"):
                user_message += f"- Specific concerns: {', '.join(context['key_issues'])}\n"
            if context.get("intake_brief"):
                user_message += f"\n## Client's Intake Brief (tailor your review to this):\n{context['intake_brief']}\n\nFocus your review on the client's stated objective, deadline, best-case and worst-case scenarios. Prioritize findings that directly affect what they care about.\n"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=800,
        )

        result = json.loads(response.choices[0].message.content)
        result["agent"] = "contract_review"
        result["status"] = "reviewed"
        return result

    async def compare_contracts(self, contract_a: str, contract_b: str) -> dict:
        """Compare two versions of a contract and highlight differences."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a contract comparison expert. Compare two contract versions and identify all meaningful differences. Return JSON with: {\"differences\": [{\"section\": \"\", \"version_a\": \"\", \"version_b\": \"\", \"impact\": \"HIGH/MEDIUM/LOW\", \"recommendation\": \"\"}], \"summary\": \"\"}"},
                {"role": "user", "content": f"CONTRACT A:\n{contract_a}\n\n---\n\nCONTRACT B:\n{contract_b}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        result = json.loads(response.choices[0].message.content)
        result["agent"] = "contract_review"
        result["status"] = "compared"
        return result
