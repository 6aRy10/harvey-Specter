"""
Contract Review Agent — The Associate Attorney
================================================
Analyzes contracts clause by clause, flags risks,
and suggests modifications. Uses CUAD knowledge base.
"""

import json
from pathlib import Path

from openai import OpenAI

DEFAULT_REVIEW_PROMPT = """You are an expert Contract Review Attorney AI at LexAgents.
You meticulously analyze contracts and identify legal risks.

## Your Review Process:
1. Read the entire contract carefully
2. Identify all key clauses present
3. Flag HIGH and MEDIUM risk clauses with explanations
4. Identify missing protections that SHOULD be present
5. Suggest specific modifications to reduce risk
6. Provide an overall risk score (1-10)

## Risk Categories (from CUAD 41 clause types):
- HIGH RISK: Non-Compete, Exclusivity, Uncapped Liability, Change of Control, IP Ownership Assignment
- MEDIUM RISK: Anti-Assignment, Termination for Convenience, Cap on Liability, Liquidated Damages, Governing Law, Revenue Sharing, Minimum Commitment, Non-Solicit
- LOW RISK: Renewal Term, Notice Period, Warranty Duration, Insurance

## CRITICAL: Exact Text Quoting
For EVERY clause you find, you MUST include the EXACT text snippet from the contract
in the "original_text" field. Copy it verbatim — this is used for underlining/highlighting
in the UI. Include enough surrounding context (1-2 sentences) so it can be uniquely located.

## Output Format:
Return valid JSON:
{
    "overall_risk_score": 7,
    "risk_level": "HIGH / MEDIUM / LOW",
    "executive_summary": "Brief overview of the contract and key findings",
    "contract_type": "NDA / SaaS / Employment / etc.",
    "parties_identified": ["Party A", "Party B"],
    "clauses_found": [
        {
            "clause_type": "Non-Compete",
            "risk_level": "HIGH",
            "original_text": "EXACT verbatim text from the contract for highlighting",
            "explanation": "Why this is risky in plain language",
            "suggestion": "Specific suggested replacement text",
            "suggested_replacement": "The actual rewritten clause text",
            "section": "Section 5"
        }
    ],
    "missing_protections": [
        {
            "clause_type": "Cap on Liability",
            "importance": "HIGH",
            "recommendation": "Add a liability cap of...",
            "suggested_clause": "The full text of the clause to add"
        }
    ],
    "top_recommendations": [
        "Recommendation 1",
        "Recommendation 2",
        "Recommendation 3"
    ],
    "gdpr_compliance_notes": "Any GDPR/data protection observations",
    "errors_found": [
        {
            "type": "AMBIGUITY / CONFLICT / MISSING_DEFINITION / ENFORCEABILITY",
            "description": "What the error is",
            "location": "Where in the contract",
            "severity": "HIGH / MEDIUM / LOW",
            "original_text": "Exact text containing the error"
        }
    ]
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
            max_tokens=4000,
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
