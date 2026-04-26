"""
LexAgents Data Pipeline Runner
===============================
Run this script to download, process, and index all legal datasets.
This is Step 1 of building the AI Law Firm.

Usage:
    1. Copy .env.example to .env and add your OPENAI_API_KEY
    2. Run: python run_pipeline.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]LexAgents — AI Law Firm Data Pipeline[/]\n"
        "Building your legal knowledge base...",
        border_style="cyan",
    ))

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]ERROR: OPENAI_API_KEY not set![/]")
        console.print("Copy .env.example to .env and add your OpenAI API key.")
        sys.exit(1)

    output_dir = "data/processed"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Process CUAD Dataset ───
    console.print("\n[bold]Step 1/4: Processing CUAD dataset...[/]")
    try:
        from data_pipeline.cuad_processor import run_cuad_pipeline
        clause_examples, contracts, review_prompt = run_cuad_pipeline(output_dir)
        console.print("[green]✓ CUAD processing complete[/]\n")
    except Exception as e:
        console.print(f"[red]CUAD processing failed: {e}[/]")
        console.print("[yellow]Continuing with other datasets...[/]\n")

    # ─── Step 2: Process German Law Dataset ───
    console.print("[bold]Step 2/4: Processing German Law dataset...[/]")
    try:
        from data_pipeline.german_law_processor import run_german_law_pipeline
        german_chunks = run_german_law_pipeline(output_dir)
        console.print("[green]✓ German Law processing complete[/]\n")
    except Exception as e:
        console.print(f"[red]German Law processing failed: {e}[/]")
        console.print("[yellow]Continuing...[/]\n")

    # ─── Step 3: Save OpenCLaw Templates ───
    console.print("[bold]Step 3/4: Saving contract templates...[/]")
    try:
        from data_pipeline.openclaw_templates import save_templates
        catalog = save_templates(output_dir)
        console.print("[green]✓ Templates saved[/]\n")
    except Exception as e:
        console.print(f"[red]Template saving failed: {e}[/]")

    # ─── Step 4: Build Vector Store ───
    console.print("[bold]Step 4/4: Building vector store (RAG index)...[/]")
    try:
        from data_pipeline.vector_store import build_vector_store
        chroma_client = build_vector_store(output_dir)
        console.print("[green]✓ Vector store built[/]\n")
    except Exception as e:
        console.print(f"[red]Vector store building failed: {e}[/]")

    # ─── Summary ───
    console.print(Panel.fit(
        "[bold green]Pipeline Complete![/]\n\n"
        "Your data is ready in:\n"
        f"  📁 {output_dir}/  — Processed JSON files\n"
        f"  📁 data/vector_store/  — ChromaDB vector index\n\n"
        "Next step: Run the backend server\n"
        "  [cyan]python -m uvicorn backend.main:app --reload[/]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
