"""
Drafting Agent — The Senior Associate
=======================================
Generates contracts from OpenCLaw templates and custom requirements.
Can draft from scratch or modify existing templates.
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data_pipeline.openclaw_templates import (
    BUILT_IN_TEMPLATES,
    fill_template,
    get_template,
    get_template_catalog,
)

DRAFTING_SYSTEM_PROMPT = """You are a Senior Contract Drafting Attorney at LexAgents, an AI law firm.
You draft precise, enforceable legal contracts.

## Available Templates:
{template_list}

## Your Drafting Process:
1. Understand the client's requirements
2. Select the most appropriate template (or draft from scratch)
3. Fill in all required variables
4. Add custom clauses based on specific needs
5. Ensure jurisdiction-appropriate language
6. Include proper GDPR/data protection clauses for EU contracts

## When using a template:
- Fill ALL template variables with appropriate values
- Add any additional clauses the client needs
- Modify standard language for the specific jurisdiction

## When drafting from scratch:
- Follow standard legal drafting conventions
- Include all essential clauses for the contract type
- Be precise and unambiguous

## Output Format:
Return valid JSON:
{{
    "template_used": "template_id or 'custom'",
    "contract_title": "Title of the contract",
    "contract_text": "The full contract text in Markdown format",
    "variables_used": {{"var_name": "value"}},
    "custom_clauses_added": ["Description of any extra clauses"],
    "jurisdiction_notes": "Any jurisdiction-specific considerations",
    "review_suggestions": "Suggestions for the review agent to verify"
}}
"""


class DraftingAgent:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.model = "gpt-4o-mini"
        self.templates = get_template_catalog()

    def _build_system_prompt(self) -> str:
        """Build system prompt with available templates."""
        template_list = ""
        for t in self.templates:
            template_list += f"- **{t['id']}**: {t['name']} ({t['jurisdiction']}) — Variables: {', '.join(t['variables'][:5])}...\n"

        return DRAFTING_SYSTEM_PROMPT.format(template_list=template_list)

    async def draft_contract(self, requirements: str, context: dict = None) -> dict:
        """
        Draft a contract based on requirements.
        
        Args:
            requirements: Natural language description of what the client needs
            context: Optional context from intake agent
        """
        user_message = f"Client's contract requirements:\n{requirements}\n"

        if context:
            if context.get("contract_type"):
                user_message += f"\nRequested contract type: {context['contract_type']}"
            if context.get("jurisdiction"):
                user_message += f"\nJurisdiction: {context['jurisdiction']}"
            if context.get("parties"):
                user_message += f"\nParties: {', '.join(context['parties'])}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=6000,
        )

        result = json.loads(response.choices[0].message.content)
        result["agent"] = "drafting"
        result["status"] = "drafted"
        return result

    async def modify_contract(self, contract_text: str, modifications: str) -> dict:
        """Modify an existing contract based on requested changes."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a contract modification expert. Apply the requested changes to the contract while maintaining legal consistency. Return JSON: {\"modified_contract\": \"full modified text\", \"changes_made\": [\"change 1\", \"change 2\"], \"warnings\": [\"any concerns about the changes\"]}"},
                {"role": "user", "content": f"ORIGINAL CONTRACT:\n{contract_text}\n\nREQUESTED MODIFICATIONS:\n{modifications}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=6000,
        )

        result = json.loads(response.choices[0].message.content)
        result["agent"] = "drafting"
        result["status"] = "modified"
        return result

    def list_templates(self) -> list[dict]:
        """Return available contract templates."""
        return self.templates
