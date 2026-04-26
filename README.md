# LexAgents — Autonomous AI Law Firm

A scalable, modular AI agent platform that operates as a full-service legal department.

## Architecture

| Agent | Role | Real-World Equivalent |
|-------|------|----------------------|
| **Intake Agent** | Classifies legal requests, extracts entities | Reception / Paralegal |
| **Contract Review Agent** | Analyzes contracts, flags risky clauses | Associate Attorney |
| **Legal Research Agent** | RAG-powered legal research (DE + US law) | Research Librarian |
| **Drafting Agent** | Generates contracts from templates | Senior Associate |
| **Orchestrator** | Routes work, tracks status, QA | Managing Partner |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
copy .env.example .env
# Edit .env and add your key

# 3. Run the data pipeline
python run_pipeline.py

# 4. Start the backend
python -m uvicorn backend.main:app --reload --port 8000

# 5. Start the frontend
cd frontend && npm install && npm run dev
```

## Data Sources

- **CUAD** (Atticus Project) — 510 annotated commercial contracts, 41 clause types
- **German Law Dataset** (Zenodo) — German legal texts for research RAG
- **OpenCLaw Templates** — Open-source contract templates (NDA, SaaS, Employment, DPA)
- **Liquid Legal Institute** — Legal ontologies and text analytics

## Tech Stack

- **Backend**: Python, FastAPI, OpenAI Agents SDK
- **Frontend**: Next.js, Tailwind CSS, shadcn/ui
- **Vector Store**: ChromaDB with OpenAI embeddings
- **LLM**: GPT-4o via OpenAI API

## License

Open Source — MIT License
