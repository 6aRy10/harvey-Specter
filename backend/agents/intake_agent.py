"""
Intake Agent — The Reception Desk
==================================
Classifies incoming legal requests and extracts key entities.
Routes to the appropriate specialist agent.
"""

import json
from openai import OpenAI

INTAKE_SYSTEM_PROMPT = """You are the Intake Specialist at LexAgents, an AI law firm. 
Your job is to receive a client's legal request and:

1. CLASSIFY it into one or more categories:
   - CONTRACT_REVIEW: Client wants an existing contract analyzed for risks
   - CONTRACT_DRAFTING: Client needs a new contract created
   - LEGAL_RESEARCH: Client has a legal question requiring research
   - COMPLIANCE_CHECK: Client needs regulatory compliance verification
   - GENERAL_COUNSEL: General legal advice or consultation

2. EXTRACT key entities:
   - parties: Names of parties involved
   - jurisdiction: Applicable legal jurisdiction (country/state)
   - contract_type: Type of contract if applicable (NDA, SaaS, Employment, etc.)
   - urgency: LOW, MEDIUM, HIGH
   - key_issues: List of specific legal issues mentioned
   - industry: Industry sector if identifiable

3. GENERATE a brief case summary

You MUST respond in valid JSON format with this exact structure:
{
    "classification": ["PRIMARY_CATEGORY", "SECONDARY_CATEGORY"],
    "entities": {
        "parties": ["Party A", "Party B"],
        "jurisdiction": "Germany / US / EU / etc.",
        "contract_type": "NDA / SaaS / Employment / etc.",
        "urgency": "LOW / MEDIUM / HIGH",
        "key_issues": ["issue 1", "issue 2"],
        "industry": "Technology / Finance / etc."
    },
    "case_summary": "Brief description of what the client needs",
    "recommended_agents": ["contract_review", "legal_research"],
    "follow_up_questions": ["Any clarifying questions if needed"]
}
"""


class IntakeAgent:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.model = "gpt-4o-mini"

    async def process(self, user_request: str) -> dict:
        """
        Process an incoming legal request.
        Returns structured classification and routing info.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INTAKE_SYSTEM_PROMPT},
                {"role": "user", "content": user_request},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        result = json.loads(response.choices[0].message.content)
        result["agent"] = "intake"
        result["status"] = "classified"
        return result
