#!/usr/bin/env python3
"""Searchable archive of the ported Sesame corpus.

Full-text search over the derived corpus using SQLite FTS5, which is compiled
into the stdlib `sqlite3` here - no daemon, no embedding model, no VRAM. At 41
chunks that is not a compromise: a vector index doesn't beat brute force until
five or six figures, and BM25 wins outright on the queries this corpus actually
gets (a project name, a person, an acronym), because names and technical terms are
exactly what embeddings blur.

Embeddings can be added later alongside this without replacing it - hybrid
keyword+vector is the standard answer for conversational recall. If that
happens, the embedding model must run on CPU: sentence-transformers grabs the
ROCm device by default, and that is VRAM CSM needs.

Build:   python archive.py --build
Search:  python archive.py "that thing we discussed"
"""
import argparse
import json
import os
import re
import sqlite3

_HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(_HERE, "memory_corpus.jsonl")
DB_FILE = os.path.join(_HERE, "memory_archive.db")

# Tiers that the default search path will never return. `restricted` is
# unpopulated by design (see the ported restricted.md); the guard exists so that
# if it is ever populated, it cannot leak through this path by accident.
BLOCKED_TIERS = {"restricted"}

COLUMNS = ["id", "text", "topic", "tier", "doc_type", "period",
           "year", "month", "salience", "trust_weight", "source_file"]

SCHEMA = """
CREATE VIRTUAL TABLE chunks USING fts5(
    id UNINDEXED, text, topic,
    tier UNINDEXED, doc_type UNINDEXED, period UNINDEXED,
    year UNINDEXED, month UNINDEXED, salience UNINDEXED,
    trust_weight UNINDEXED, source_file UNINDEXED,
    tokenize = "porter unicode61"
);
"""

# FTS5 treats punctuation as syntax, so a natural-language query like
# "what about that conversation?" is a syntax error rather than a search.
# Reduce to bare terms and OR them; bm25 handles the ranking from there.
_TERM = re.compile(r"[A-Za-z0-9_]+")
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is",
         "was", "what", "about", "did", "we", "i", "you", "he", "she", "it",
         "that", "this", "with", "at", "from", "my", "our", "your"}


def to_match_query(text):
    """A safe FTS5 MATCH expression, or None if nothing searchable remains."""
    terms = [t.lower() for t in _TERM.findall(text)]
    kept = [t for t in terms if t not in _STOP and len(t) > 1]
    if not kept:
        kept = terms
    return " OR ".join(kept) if kept else None


def build(corpus=CORPUS_FILE, db=DB_FILE):
    """(Re)build the index from the corpus JSONL. Returns the chunk count."""
    if os.path.exists(db):
        os.remove(db)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    rows, skipped = [], 0
    with open(corpus) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            md = r.get("metadata", {})
            if md.get("tier") in BLOCKED_TIERS:
                skipped += 1
                continue
            rows.append(tuple(
                str(r.get(c) if c in ("id", "text") else md.get(c, "") or "")
                for c in COLUMNS
            ))
    con.executemany(
        f"INSERT INTO chunks ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})",
        rows,
    )
    con.commit()
    con.close()
    return len(rows), skipped


def search(query, k=5, db=DB_FILE, period=None):
    """Top-k chunks by BM25. `period` optionally filters to e.g. '2026-07'."""
    match = to_match_query(query)
    if not match:
        return []
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    sql = (f"SELECT {','.join(COLUMNS)}, bm25(chunks) AS score FROM chunks "
           f"WHERE chunks MATCH ?")
    args = [match]
    if period:
        sql += " AND period = ?"
        args.append(period)
    sql += " ORDER BY score LIMIT ?"
    args.append(k)
    try:
        out = [dict(r) for r in con.execute(sql, args)]
    except sqlite3.OperationalError:
        out = []
    finally:
        con.close()
    return out


def format_hits(hits):
    """Render hits for injection into a prompt, cheapest-to-read form."""
    lines = []
    for h in hits:
        when = h.get("period") or "undated"
        lines.append(f"[{when}] {h['text']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Search the ported memory archive")
    ap.add_argument("query", nargs="*", help="search terms")
    ap.add_argument("--build", action="store_true", help="rebuild the index")
    ap.add_argument("-k", type=int, default=5, help="results to return")
    ap.add_argument("--period", help="filter to a period, e.g. 2026-07")
    args = ap.parse_args()

    if args.build:
        n, skipped = build()
        print(f"indexed {n} chunks -> {DB_FILE}" +
              (f" ({skipped} blocked-tier skipped)" if skipped else ""))
        if not args.query:
            return

    if not args.query:
        ap.error("give a query, or --build")
    hits = search(" ".join(args.query), k=args.k, period=args.period)
    if not hits:
        print("no matches")
        return
    for h in hits:
        print(f"\n[{h.get('period') or '----'}] {h['topic']}  "
              f"(trust {h['trust_weight']}, {h['doc_type']})")
        print(f"  {h['text']}")


if __name__ == "__main__":
    main()
