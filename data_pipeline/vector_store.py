"""
Vector Store Builder
====================
Takes processed chunks from CUAD + German Law datasets,
embeds them with OpenAI embeddings, and stores in ChromaDB
for fast RAG retrieval.
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from rich.console import Console
from rich.progress import track

console = Console()

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # OpenAI embedding batch limit
CHROMA_DIR = "data/vector_store"


def get_openai_client():
    """Initialize OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI embeddings."""
    # Filter empty texts
    texts = [t if t.strip() else "empty" for t in texts]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def build_vector_store(
    processed_dir="data/processed",
    chroma_dir=CHROMA_DIR,
):
    """
    Build ChromaDB vector store from all processed legal data.
    Creates two collections:
      1. 'contracts' — CUAD contract texts for contract review RAG
      2. 'legal_knowledge' — German law + legal research knowledge base
    """
    console.print("[bold cyan]═══ Building Vector Store ═══[/]")

    processed_path = Path(processed_dir)
    client_openai = get_openai_client()

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    # ─── Collection 1: Contracts (from CUAD) ───
    console.print("\n[bold blue]Building 'contracts' collection...[/]")

    contracts_file = processed_path / "cuad_contracts.json"
    if contracts_file.exists():
        with open(contracts_file, "r", encoding="utf-8") as f:
            contracts = json.load(f)

        collection = chroma_client.get_or_create_collection(
            name="contracts",
            metadata={"description": "CUAD contract texts for contract review"},
        )

        # Chunk contracts further if needed and embed
        all_ids = []
        all_texts = []
        all_metadatas = []

        for contract in contracts:
            text = contract["text"]
            # Split large contracts into chunks
            chunk_size = 1500
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                chunk_id = f"cuad_{contract['title'].replace(' ', '_')[:50]}_{i}"
                all_ids.append(chunk_id)
                all_texts.append(chunk)
                all_metadatas.append({
                    "source": "CUAD",
                    "title": contract["title"],
                    "chunk_start": i,
                })

        # Embed and store in batches
        for batch_start in track(
            range(0, len(all_texts), BATCH_SIZE),
            description="Embedding contracts",
        ):
            batch_end = min(batch_start + BATCH_SIZE, len(all_texts))
            batch_texts = all_texts[batch_start:batch_end]
            batch_ids = all_ids[batch_start:batch_end]
            batch_meta = all_metadatas[batch_start:batch_end]

            embeddings = embed_texts(client_openai, batch_texts)

            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )

        console.print(f"[green]✓ Indexed {len(all_texts)} contract chunks[/]")
    else:
        console.print("[yellow]No CUAD contracts found, skipping...[/]")

    # ─── Collection 2: Legal Knowledge (from German Law + clause examples) ───
    console.print("\n[bold blue]Building 'legal_knowledge' collection...[/]")

    collection = chroma_client.get_or_create_collection(
        name="legal_knowledge",
        metadata={"description": "German law + legal knowledge base for research"},
    )

    all_ids = []
    all_texts = []
    all_metadatas = []

    # Add German law chunks
    german_law_file = processed_path / "german_law_chunks.json"
    if german_law_file.exists():
        with open(german_law_file, "r", encoding="utf-8") as f:
            german_chunks = json.load(f)

        for chunk in german_chunks:
            all_ids.append(chunk["id"])
            all_texts.append(chunk["text"])
            all_metadatas.append({
                "source": chunk["source"],
                "dataset": chunk["dataset"],
                "chunk_index": chunk["chunk_index"],
            })

        console.print(f"  Added {len(german_chunks)} German law chunks")

    # Add CUAD clause examples as knowledge
    clause_file = processed_path / "cuad_clause_examples.json"
    if clause_file.exists():
        with open(clause_file, "r", encoding="utf-8") as f:
            clause_examples = json.load(f)

        for clause_type, examples in clause_examples.items():
            for i, example in enumerate(examples):
                chunk_id = f"clause_{clause_type.replace(' ', '_')}_{i}"
                text = f"[{clause_type}] {example}"
                all_ids.append(chunk_id)
                all_texts.append(text)
                all_metadatas.append({
                    "source": "CUAD Clause Examples",
                    "clause_type": clause_type,
                })

        console.print(f"  Added clause examples from CUAD")

    # Embed and store in batches
    if all_texts:
        for batch_start in track(
            range(0, len(all_texts), BATCH_SIZE),
            description="Embedding legal knowledge",
        ):
            batch_end = min(batch_start + BATCH_SIZE, len(all_texts))
            batch_texts = all_texts[batch_start:batch_end]
            batch_ids = all_ids[batch_start:batch_end]
            batch_meta = all_metadatas[batch_start:batch_end]

            embeddings = embed_texts(client_openai, batch_texts)

            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )

        console.print(f"[green]✓ Indexed {len(all_texts)} legal knowledge chunks[/]")
    else:
        console.print("[yellow]No legal knowledge data found to index[/]")

    console.print(f"\n[bold green]✓ Vector store built at {chroma_dir}[/]")
    return chroma_client


def search_vector_store(
    query: str,
    collection_name: str = "legal_knowledge",
    n_results: int = 5,
    chroma_dir: str = CHROMA_DIR,
):
    """
    Search the vector store. Used by agents for RAG retrieval.
    """
    client_openai = get_openai_client()
    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    collection = chroma_client.get_collection(name=collection_name)

    # Embed the query
    query_embedding = embed_texts(client_openai, [query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    formatted = []
    for i in range(len(results["ids"][0])):
        dist = results["distances"][0][i]
        formatted.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", ""),
            "metadata": results["metadatas"][0][i],
            "distance": dist,
            "similarity": round(max(0, 1 - dist / 2), 3),  # cosine distance → similarity
        })

    return formatted


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB text limit
MAX_CHUNKS_PER_DOC = 5000  # safety cap
ALLOWED_COLLECTION_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


def _sanitize_collection_name(name: str) -> str:
    """Ensure collection name is safe for ChromaDB."""
    name = name.strip().lower().replace(" ", "_")
    name = "".join(c for c in name if c in ALLOWED_COLLECTION_CHARS)
    if not name or len(name) < 3:
        name = "firm_policies"
    return name[:63]  # ChromaDB max collection name length


def ingest_document(
    text: str,
    filename: str,
    collection_name: str = "firm_policies",
    chroma_dir: str = CHROMA_DIR,
    chunk_size: int = 1200,
):
    """
    Ingest a single document into the vector store.
    Chunks it, embeds it, and stores in the specified collection.
    Returns number of chunks created.
    """
    import re, time, hashlib

    # Validate inputs
    if not text or not isinstance(text, str):
        return 0
    if len(text) > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {len(text)} bytes (max {MAX_FILE_SIZE})")
    if not filename:
        filename = "unnamed_document.txt"

    collection_name = _sanitize_collection_name(collection_name)

    try:
        client_openai = get_openai_client()
    except Exception as e:
        raise RuntimeError(f"OpenAI client init failed: {e}")

    try:
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Collection: {collection_name}"},
        )
    except Exception as e:
        raise RuntimeError(f"ChromaDB init failed: {e}")

    # Chunk the document
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    if not chunks:
        return 0

    # Cap chunks to prevent runaway embedding costs
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        chunks = chunks[:MAX_CHUNKS_PER_DOC]

    # Create deterministic IDs (prevents duplicates on re-upload)
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)[:50]
    doc_hash = hashlib.md5(text[:2000].encode("utf-8", errors="ignore")).hexdigest()[:8]
    ids = [f"{safe_name}_{doc_hash}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "collection": collection_name, "chunk_index": i} for i in range(len(chunks))]

    # Embed and store with retry
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(chunks))
        batch_texts = chunks[batch_start:batch_end]

        # Retry embedding up to 3 times
        embeddings = None
        for attempt in range(3):
            try:
                embeddings = embed_texts(client_openai, batch_texts)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Embedding failed after 3 attempts: {e}")
                time.sleep(1 * (attempt + 1))

        # Use upsert to handle duplicate IDs gracefully
        try:
            collection.upsert(
                ids=ids[batch_start:batch_end],
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=metadatas[batch_start:batch_end],
            )
        except Exception as e:
            raise RuntimeError(f"ChromaDB upsert failed at batch {batch_start}: {e}")

    return len(chunks)


def ingest_folder(
    folder_path: str,
    collection_name: str = "firm_policies",
    chroma_dir: str = CHROMA_DIR,
):
    """
    Ingest all supported files from a folder into the vector store.
    Supports: .txt, .md, .pdf, .json, .csv, .docx
    Returns dict with results per file.
    """
    folder = Path(folder_path)
    if not folder.exists():
        return {"error": f"Folder not found: {folder_path}"}
    if not folder.is_dir():
        return {"error": f"Not a directory: {folder_path}"}

    collection_name = _sanitize_collection_name(collection_name)
    results = {"files_processed": 0, "total_chunks": 0, "total_files": 0, "files": [], "errors": []}
    extensions = {".txt", ".md", ".pdf", ".json", ".csv", ".docx"}
    max_files = 500  # safety cap

    files_found = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in extensions)
    results["total_files"] = len(files_found)

    for filepath in files_found[:max_files]:
        try:
            # Size check
            if filepath.stat().st_size > MAX_FILE_SIZE:
                results["errors"].append({"name": filepath.name, "error": "File too large (>50MB)"})
                continue
            if filepath.stat().st_size == 0:
                results["errors"].append({"name": filepath.name, "error": "Empty file"})
                continue

            if filepath.suffix.lower() == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(str(filepath))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                except ImportError:
                    results["errors"].append({"name": filepath.name, "error": "pypdf not installed"})
                    continue
                except Exception as e:
                    results["errors"].append({"name": filepath.name, "error": f"PDF read error: {e}"})
                    continue
            else:
                try:
                    text = filepath.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    results["errors"].append({"name": filepath.name, "error": f"Read error: {e}"})
                    continue

            if not text.strip():
                results["errors"].append({"name": filepath.name, "error": "No text extracted"})
                continue

            n_chunks = ingest_document(text, filepath.name, collection_name, chroma_dir)
            results["files_processed"] += 1
            results["total_chunks"] += n_chunks
            results["files"].append({"name": filepath.name, "chunks": n_chunks, "size": len(text)})
        except Exception as e:
            results["errors"].append({"name": filepath.name, "error": str(e)})

    return results


def list_collections(chroma_dir: str = CHROMA_DIR):
    """List all collections and their document counts."""
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    collections = chroma_client.list_collections()
    result = []
    for col in collections:
        c = chroma_client.get_collection(col.name if hasattr(col, 'name') else col)
        name = col.name if hasattr(col, 'name') else col
        result.append({"name": name, "count": c.count()})
    return result


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    build_vector_store()

    # Test search
    console.print("\n[bold cyan]Testing search...[/]")
    results = search_vector_store("non-compete clause restrictions")
    for r in results:
        console.print(f"  [{r['distance']:.3f}] {r['text'][:100]}...")
