"""
Demo Scenarios for LexAgents
=============================
Three killer demo scenarios to showcase during the hackathon presentation.
Run each one to generate impressive output.

Usage: python demo/demo_scenarios.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

load_dotenv()

from openai import OpenAI
from backend.agents.orchestrator import Orchestrator

console = Console()


async def demo_1_contract_review():
    """Demo 1: Review the sample NDA — find risky clauses."""
    console.print(Panel.fit(
        "[bold red]DEMO 1: Contract Review[/]\n"
        "Upload a sample NDA → AI identifies 5+ risky clauses → suggests fixes",
        border_style="red",
    ))

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    orchestrator = Orchestrator(client)

    # Load sample NDA
    nda_path = Path(__file__).parent / "sample_nda.txt"
    nda_text = nda_path.read_text(encoding="utf-8")

    result = await orchestrator.create_matter(
        client_request="Please review this NDA between TechCorp GmbH (Germany) and DataFlow Inc (US). "
                       "I'm the Receiving Party and want to know what risks I'm taking on.",
        contract_text=nda_text,
    )

    console.print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


async def demo_2_draft_saas():
    """Demo 2: Draft a SaaS agreement for a German client."""
    console.print(Panel.fit(
        "[bold blue]DEMO 2: Contract Drafting[/]\n"
        "Draft a SaaS agreement → Auto-review for quality → GDPR compliance",
        border_style="blue",
    ))

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    orchestrator = Orchestrator(client)

    result = await orchestrator.create_matter(
        client_request=(
            "I need a SaaS Service Agreement for my AI analytics platform. "
            "The provider is LexTech GmbH based in Munich, Germany. "
            "The customer is a UK-based law firm, Baker & Associates LLP. "
            "The service costs €500/month with a 12-month initial term. "
            "We need GDPR compliance, 99.9% uptime SLA, and a data processing agreement. "
            "Governing law should be German law."
        ),
    )

    console.print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


async def demo_3_legal_research():
    """Demo 3: Research GDPR implications."""
    console.print(Panel.fit(
        "[bold green]DEMO 3: Legal Research[/]\n"
        "GDPR analysis → German law citations → Practical recommendations",
        border_style="green",
    ))

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    orchestrator = Orchestrator(client)

    result = await orchestrator.create_matter(
        client_request=(
            "What are the GDPR implications if our SaaS platform (hosted in AWS Frankfurt) "
            "processes personal data of EU citizens and transfers it to a sub-processor "
            "in the United States? We need to understand: "
            "1) What legal basis do we need for the transfer? "
            "2) What safeguards are required post-Schrems II? "
            "3) What does the German BDSG add on top of GDPR? "
            "4) What penalties could we face for non-compliance?"
        ),
    )

    console.print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


async def main():
    console.print(Panel.fit(
        "[bold cyan]LexAgents — Demo Scenarios[/]\n"
        "Running 3 demo scenarios to showcase the AI Law Firm",
        border_style="cyan",
    ))

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Set OPENAI_API_KEY in .env first![/]")
        return

    # Run demos
    console.print("\n" + "=" * 60)
    r1 = await demo_1_contract_review()

    console.print("\n" + "=" * 60)
    r2 = await demo_2_draft_saas()

    console.print("\n" + "=" * 60)
    r3 = await demo_3_legal_research()

    # Save results
    output_dir = Path("demo/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, result in [("review", r1), ("draft", r2), ("research", r3)]:
        with open(output_dir / f"demo_{name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    console.print(Panel.fit(
        "[bold green]All demos complete![/]\n"
        "Results saved to demo/results/",
        border_style="green",
    ))


if __name__ == "__main__":
    asyncio.run(main())
