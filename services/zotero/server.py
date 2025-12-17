#!/usr/bin/env python3
"""
Zotero REST API Server
======================
REST API wrapper around Zotero translation-server with additional features:
- Paper metadata extraction from URLs/DOIs
- BibTeX/CSL-JSON export
- PDF storage management
- Integration with Khoj for indexing
"""

import os
import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
import requests
import bibtexparser

app = Flask(__name__)

# Configuration
TRANSLATION_SERVER = "http://localhost:1969"
DATA_DIR = Path(os.environ.get("ZOTERO_DATA_DIR", "/data/zotero"))
STORAGE_DIR = Path(os.environ.get("ZOTERO_STORAGE_DIR", "/data/storage"))
DB_PATH = DATA_DIR / "papers.db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def init_db():
    """Initialize SQLite database for paper metadata."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            year INTEGER,
            doi TEXT,
            url TEXT,
            abstract TEXT,
            bibtex TEXT,
            csl_json TEXT,
            pdf_path TEXT,
            tags TEXT,
            notes TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)
    """)
    conn.commit()
    conn.close()


def generate_paper_id(title: str, doi: str = None) -> str:
    """Generate unique paper ID."""
    if doi:
        return hashlib.sha256(doi.encode()).hexdigest()[:12]
    return hashlib.sha256(title.encode()).hexdigest()[:12]


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "zotero-api"})


@app.route("/api/translate", methods=["POST"])
def translate_url():
    """
    Translate URL to paper metadata using Zotero translation-server.

    Request body:
        {"url": "https://arxiv.org/abs/..."}

    Returns:
        Paper metadata in CSL-JSON format
    """
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "URL required"}), 400

    try:
        # Call translation-server
        resp = requests.post(
            f"{TRANSLATION_SERVER}/web",
            data=url,
            headers={"Content-Type": "text/plain"},
            timeout=30
        )

        if resp.status_code == 200:
            items = resp.json()
            return jsonify({"success": True, "items": items})
        else:
            return jsonify({"error": f"Translation failed: {resp.status_code}"}), 500

    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/papers", methods=["GET"])
def list_papers():
    """List all papers in the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Optional filters
    year = request.args.get("year")
    tag = request.args.get("tag")
    search = request.args.get("q")

    query = "SELECT * FROM papers WHERE 1=1"
    params = []

    if year:
        query += " AND year = ?"
        params.append(int(year))

    if tag:
        query += " AND tags LIKE ?"
        params.append(f"%{tag}%")

    if search:
        query += " AND (title LIKE ? OR abstract LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])

    query += " ORDER BY created_at DESC"

    cursor.execute(query, params)
    papers = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({"papers": papers, "count": len(papers)})


@app.route("/api/papers", methods=["POST"])
def add_paper():
    """
    Add a paper to the database.

    Request body:
        {
            "url": "https://...",  # Optional - will translate
            "title": "Paper Title",
            "authors": "Author 1, Author 2",
            "year": 2024,
            "doi": "10.1234/...",
            "abstract": "...",
            "tags": ["ml", "transformers"]
        }
    """
    data = request.get_json()

    # If URL provided, try to translate first
    if "url" in data and not data.get("title"):
        try:
            resp = requests.post(
                f"{TRANSLATION_SERVER}/web",
                data=data["url"],
                headers={"Content-Type": "text/plain"},
                timeout=30
            )
            if resp.status_code == 200:
                items = resp.json()
                if items:
                    item = items[0]
                    data["title"] = item.get("title", data.get("title", "Untitled"))
                    data["authors"] = ", ".join(
                        f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
                        for c in item.get("creators", [])
                    )
                    data["year"] = item.get("date", "")[:4] if item.get("date") else None
                    data["doi"] = item.get("DOI")
                    data["abstract"] = item.get("abstractNote")
                    data["csl_json"] = json.dumps(item)
        except Exception:
            pass

    if not data.get("title"):
        return jsonify({"error": "Title required"}), 400

    paper_id = generate_paper_id(data.get("title"), data.get("doi"))
    now = datetime.utcnow().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT OR REPLACE INTO papers
            (id, title, authors, year, doi, url, abstract, bibtex, csl_json, tags, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper_id,
            data.get("title"),
            data.get("authors"),
            data.get("year"),
            data.get("doi"),
            data.get("url"),
            data.get("abstract"),
            data.get("bibtex"),
            data.get("csl_json"),
            json.dumps(data.get("tags", [])),
            data.get("notes"),
            now,
            now
        ))
        conn.commit()

        return jsonify({"success": True, "id": paper_id})

    except sqlite3.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/papers/<paper_id>", methods=["GET"])
def get_paper(paper_id):
    """Get a specific paper by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Paper not found"}), 404

    return jsonify(dict(row))


@app.route("/api/papers/<paper_id>", methods=["DELETE"])
def delete_paper(paper_id):
    """Delete a paper."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()

    if deleted:
        return jsonify({"success": True})
    return jsonify({"error": "Paper not found"}), 404


@app.route("/api/export/bibtex", methods=["GET"])
def export_bibtex():
    """Export all papers as BibTeX."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers")
    papers = cursor.fetchall()
    conn.close()

    entries = []
    for paper in papers:
        if paper["bibtex"]:
            entries.append(paper["bibtex"])
        else:
            # Generate BibTeX entry
            entry = f"""@article{{{paper["id"]},
  title = {{{paper["title"]}}},
  author = {{{paper["authors"] or "Unknown"}}},
  year = {{{paper["year"] or "n.d."}}},
  doi = {{{paper["doi"] or ""}}},
}}"""
            entries.append(entry)

    return "\n\n".join(entries), 200, {"Content-Type": "text/plain"}


@app.route("/api/export/markdown", methods=["GET"])
def export_markdown():
    """Export papers as Markdown for Obsidian."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers ORDER BY year DESC")
    papers = cursor.fetchall()
    conn.close()

    lines = ["# Paper Library\n"]

    current_year = None
    for paper in papers:
        year = paper["year"] or "Unknown"
        if year != current_year:
            lines.append(f"\n## {year}\n")
            current_year = year

        tags = json.loads(paper["tags"]) if paper["tags"] else []
        tag_str = " ".join(f"#{t}" for t in tags)

        lines.append(f"### {paper['title']}")
        lines.append(f"**Authors:** {paper['authors'] or 'Unknown'}")
        if paper["doi"]:
            lines.append(f"**DOI:** [{paper['doi']}](https://doi.org/{paper['doi']})")
        if tag_str:
            lines.append(f"**Tags:** {tag_str}")
        if paper["abstract"]:
            lines.append(f"\n> {paper['abstract'][:500]}...")
        lines.append("")

    return "\n".join(lines), 200, {"Content-Type": "text/markdown"}


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get library statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM papers")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT year, COUNT(*) FROM papers WHERE year IS NOT NULL GROUP BY year ORDER BY year DESC")
    by_year = dict(cursor.fetchall())

    conn.close()

    return jsonify({
        "total_papers": total,
        "by_year": by_year
    })


# Initialize database on startup
init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8085, debug=False)
