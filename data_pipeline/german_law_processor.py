"""
German Law Dataset Processor
=============================
Downloads and processes the German law dataset from Zenodo.
Chunks legal texts and prepares them for RAG embedding + vector search.

Source: https://zenodo.org/records/19481365
"""

import json
import os
import re
import zipfile
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import track

console = Console()

ZENODO_RECORD_ID = "19481365"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def get_zenodo_download_urls():
    """Fetch file download URLs from Zenodo API."""
    console.print(f"[bold green]Fetching Zenodo record {ZENODO_RECORD_ID}...[/]")
    resp = requests.get(ZENODO_API_URL)
    resp.raise_for_status()
    record = resp.json()

    files = []
    for f in record.get("files", []):
        files.append({
            "filename": f["key"],
            "url": f["links"]["self"],
            "size": f["size"],
        })
        console.print(f"  Found: {f['key']} ({f['size'] / 1024 / 1024:.1f} MB)")

    return files, record.get("metadata", {})


def download_zenodo_files(files, download_dir="data/raw/german_law"):
    """Download files from Zenodo."""
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for f in files:
        target = download_path / f["filename"]
        if target.exists():
            console.print(f"  [yellow]Skipping {f['filename']} (already exists)[/]")
            downloaded.append(target)
            continue

        console.print(f"  [blue]Downloading {f['filename']}...[/]")
        resp = requests.get(f["url"], stream=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        with open(target, "wb") as out:
            for chunk in resp.iter_content(chunk_size=8192):
                out.write(chunk)

        downloaded.append(target)
        console.print(f"  [green]✓ Downloaded {f['filename']}[/]")

    return downloaded


def extract_files(downloaded_files, extract_dir="data/raw/german_law/extracted"):
    """Extract zip/gz files if present."""
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)

    extracted = []
    for f in downloaded_files:
        if f.suffix == ".zip":
            console.print(f"  [blue]Extracting {f.name}...[/]")
            with zipfile.ZipFile(f, "r") as z:
                z.extractall(extract_path)
            extracted.extend(extract_path.rglob("*"))
        else:
            extracted.append(f)

    return extracted


def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks for embedding.
    Uses sentence boundaries when possible.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap // 5 :]) if len(words) > overlap // 5 else ""
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def process_legal_texts(files, output_dir="data/processed"):
    """
    Process downloaded legal text files into chunks ready for embedding.
    Handles JSON, TXT, CSV, and Markdown files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    doc_id = 0

    for filepath in files:
        filepath = Path(filepath)
        if filepath.is_dir():
            continue
        if filepath.suffix not in [".json", ".txt", ".md", ".csv", ".jsonl"]:
            continue

        console.print(f"  [blue]Processing {filepath.name}...[/]")

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            console.print(f"  [red]Error reading {filepath.name}: {e}[/]")
            continue

        # Handle different formats
        if filepath.suffix == ".json":
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        text = ""
                        if isinstance(item, dict):
                            # Try common field names
                            for key in ["text", "content", "body", "abstract", "description", "law_text"]:
                                if key in item:
                                    text = str(item[key])
                                    break
                            if not text:
                                text = json.dumps(item, ensure_ascii=False)
                        else:
                            text = str(item)

                        if text.strip():
                            chunks = chunk_text(text)
                            for i, chunk in enumerate(chunks):
                                all_chunks.append({
                                    "id": f"german_law_{doc_id}_{i}",
                                    "text": chunk,
                                    "source": filepath.name,
                                    "dataset": "German Law (Zenodo)",
                                    "chunk_index": i,
                                    "total_chunks": len(chunks),
                                })
                            doc_id += 1
                elif isinstance(data, dict):
                    text = json.dumps(data, ensure_ascii=False)
                    chunks = chunk_text(text)
                    for i, chunk in enumerate(chunks):
                        all_chunks.append({
                            "id": f"german_law_{doc_id}_{i}",
                            "text": chunk,
                            "source": filepath.name,
                            "dataset": "German Law (Zenodo)",
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        })
                    doc_id += 1
            except json.JSONDecodeError:
                pass

        elif filepath.suffix == ".jsonl":
            for line_num, line in enumerate(content.split("\n")):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = ""
                    if isinstance(item, dict):
                        for key in ["text", "content", "body", "abstract", "description", "law_text"]:
                            if key in item:
                                text = str(item[key])
                                break
                    if text.strip():
                        chunks = chunk_text(text)
                        for i, chunk in enumerate(chunks):
                            all_chunks.append({
                                "id": f"german_law_{doc_id}_{i}",
                                "text": chunk,
                                "source": filepath.name,
                                "dataset": "German Law (Zenodo)",
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                            })
                        doc_id += 1
                except json.JSONDecodeError:
                    continue

        else:  # .txt, .md, .csv
            if content.strip():
                chunks = chunk_text(content)
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "id": f"german_law_{doc_id}_{i}",
                        "text": chunk,
                        "source": filepath.name,
                        "dataset": "German Law (Zenodo)",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    })
                doc_id += 1

    console.print(f"[green]✓ Created {len(all_chunks)} chunks from {doc_id} documents[/]")

    # Save chunks
    with open(output_path / "german_law_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    return all_chunks


def run_german_law_pipeline(output_dir="data/processed"):
    """Run the full German law processing pipeline."""
    console.print("[bold cyan]═══ German Law Dataset Pipeline ═══[/]")

    # Step 1: Get download URLs
    files, metadata = get_zenodo_download_urls()
    if not files:
        console.print("[red]No files found in Zenodo record. Using fallback approach.[/]")
        return []

    console.print(f"[green]Dataset: {metadata.get('title', 'Unknown')}[/]")

    # Step 2: Download
    downloaded = download_zenodo_files(files)

    # Step 3: Extract if needed
    all_files = extract_files(downloaded)

    # Step 4: Process into chunks
    chunks = process_legal_texts(all_files, output_dir)

    return chunks


if __name__ == "__main__":
    run_german_law_pipeline()
