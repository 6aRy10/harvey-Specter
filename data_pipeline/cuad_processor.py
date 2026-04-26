"""
CUAD Dataset Processor
======================
Downloads and processes the Atticus Project CUAD dataset from HuggingFace.
Extracts 41 clause types with examples for the Contract Review Agent.

CUAD contains 510 real commercial contracts with 13,000+ expert annotations.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()

# The 41 CUAD clause categories — these map to real legal risk areas
CUAD_CLAUSE_TYPES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
]

# Risk levels for each clause type (for the review agent)
CLAUSE_RISK_MAP = {
    "Non-Compete": "HIGH",
    "Exclusivity": "HIGH",
    "Uncapped Liability": "HIGH",
    "Change Of Control": "HIGH",
    "Ip Ownership Assignment": "HIGH",
    "Anti-Assignment": "MEDIUM",
    "Termination For Convenience": "MEDIUM",
    "Cap On Liability": "MEDIUM",
    "Liquidated Damages": "MEDIUM",
    "Non-Transferable License": "MEDIUM",
    "Governing Law": "MEDIUM",
    "Revenue/Profit Sharing": "MEDIUM",
    "Minimum Commitment": "MEDIUM",
    "No-Solicit Of Employees": "MEDIUM",
    "No-Solicit Of Customers": "MEDIUM",
    "Non-Disparagement": "LOW",
    "Renewal Term": "LOW",
    "Notice Period To Terminate Renewal": "LOW",
    "Warranty Duration": "LOW",
    "Insurance": "LOW",
}


def download_cuad():
    """Download CUAD dataset from HuggingFace."""
    console.print("[bold green]Downloading CUAD dataset from HuggingFace...[/]")
    dataset = load_dataset("theatticusproject/cuad", trust_remote_code=True)
    console.print(f"[green]✓ Downloaded {len(dataset['test'])} samples[/]")
    return dataset


def extract_clause_examples(dataset, max_examples_per_clause=5):
    """
    Extract example annotations for each of the 41 clause types.
    Returns a dict: {clause_type: [list of example text snippets]}
    """
    clause_examples = {clause: [] for clause in CUAD_CLAUSE_TYPES}

    console.print("[bold blue]Extracting clause examples...[/]")

    for sample in track(dataset["test"], description="Processing contracts"):
        context = sample.get("context", "")

        for clause_type in CUAD_CLAUSE_TYPES:
            # CUAD stores answers in a format like: answers[clause_type]
            answer_key = clause_type
            answers = sample.get("answers", {})

            if isinstance(answers, dict):
                texts = answers.get("text", [])
                if texts and len(clause_examples[clause_type]) < max_examples_per_clause:
                    for text in texts:
                        if text and text.strip():
                            clause_examples[clause_type].append(text.strip())
                            if len(clause_examples[clause_type]) >= max_examples_per_clause:
                                break

    # Count how many we found
    total = sum(len(v) for v in clause_examples.values())
    non_empty = sum(1 for v in clause_examples.values() if v)
    console.print(f"[green]✓ Extracted {total} examples across {non_empty}/{len(CUAD_CLAUSE_TYPES)} clause types[/]")

    return clause_examples


def extract_full_contracts(dataset, max_contracts=20):
    """Extract full contract texts for RAG indexing."""
    contracts = []
    seen = set()

    for sample in dataset["test"]:
        context = sample.get("context", "")
        title = sample.get("title", "Unknown Contract")

        if title not in seen and context.strip():
            seen.add(title)
            contracts.append({
                "title": title,
                "text": context.strip(),
                "source": "CUAD / Atticus Project",
            })
            if len(contracts) >= max_contracts:
                break

    console.print(f"[green]✓ Extracted {len(contracts)} unique contracts[/]")
    return contracts


def build_contract_review_prompt(clause_examples):
    """
    Build the system prompt for the Contract Review Agent
    using real CUAD examples as few-shot references.
    """
    prompt_parts = [
        "You are an expert Contract Review Attorney AI. You analyze contracts and identify risks.",
        "You check for the following 41 clause categories, flagging risks and suggesting improvements.",
        "",
        "## Clause Categories & Risk Levels",
        "",
    ]

    for clause in CUAD_CLAUSE_TYPES:
        risk = CLAUSE_RISK_MAP.get(clause, "INFO")
        examples = clause_examples.get(clause, [])

        prompt_parts.append(f"### {clause} [Risk: {risk}]")
        if examples:
            prompt_parts.append("Example from real contracts:")
            for i, ex in enumerate(examples[:2], 1):
                # Truncate long examples
                snippet = ex[:300] + "..." if len(ex) > 300 else ex
                prompt_parts.append(f'  {i}. "{snippet}"')
        prompt_parts.append("")

    prompt_parts.extend([
        "## Your Review Process:",
        "1. Read the entire contract carefully",
        "2. Identify which of the 41 clause types are present",
        "3. Flag any HIGH or MEDIUM risk clauses",
        "4. For each flagged clause, explain the risk in plain language",
        "5. Suggest specific modifications to reduce risk",
        "6. Provide an overall risk score from 1-10",
        "7. Generate an executive summary",
        "",
        "## Output Format:",
        "Return a structured JSON with:",
        '- "overall_risk_score": 1-10',
        '- "executive_summary": brief overview',
        '- "clauses_found": [{clause_type, text, risk_level, explanation, suggestion}]',
        '- "missing_protections": clauses that SHOULD be present but are not',
        '- "recommendations": top 3 action items',
    ])

    return "\n".join(prompt_parts)


def save_processed_data(output_dir: str, clause_examples: dict, contracts: list, review_prompt: str):
    """Save all processed data to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save clause examples
    with open(output_path / "cuad_clause_examples.json", "w", encoding="utf-8") as f:
        json.dump(clause_examples, f, indent=2, ensure_ascii=False)

    # Save contracts
    with open(output_path / "cuad_contracts.json", "w", encoding="utf-8") as f:
        json.dump(contracts, f, indent=2, ensure_ascii=False)

    # Save the review agent system prompt
    with open(output_path / "contract_review_prompt.txt", "w", encoding="utf-8") as f:
        f.write(review_prompt)

    # Save clause metadata
    clause_metadata = []
    for clause in CUAD_CLAUSE_TYPES:
        clause_metadata.append({
            "name": clause,
            "risk_level": CLAUSE_RISK_MAP.get(clause, "INFO"),
            "example_count": len(clause_examples.get(clause, [])),
        })
    with open(output_path / "cuad_clause_metadata.json", "w", encoding="utf-8") as f:
        json.dump(clause_metadata, f, indent=2)

    console.print(f"[bold green]✓ All CUAD data saved to {output_path}[/]")


def run_cuad_pipeline(output_dir="data/processed"):
    """Run the full CUAD processing pipeline."""
    console.print("[bold cyan]═══ CUAD Dataset Pipeline ═══[/]")

    # Step 1: Download
    dataset = download_cuad()

    # Step 2: Extract clause examples
    clause_examples = extract_clause_examples(dataset)

    # Step 3: Extract full contracts
    contracts = extract_full_contracts(dataset)

    # Step 4: Build review prompt
    review_prompt = build_contract_review_prompt(clause_examples)
    console.print(f"[green]✓ Built contract review prompt ({len(review_prompt)} chars)[/]")

    # Step 5: Save
    save_processed_data(output_dir, clause_examples, contracts, review_prompt)

    return clause_examples, contracts, review_prompt


if __name__ == "__main__":
    run_cuad_pipeline()
