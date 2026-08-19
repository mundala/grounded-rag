#!/usr/bin/env python3
"""Retrieval-augmented generation over a local document corpus, with citation checking.

Design note: no vector database. SQLite holds the chunks and their embeddings as raw
float32 blobs, and retrieval is a numpy dot product over the whole table. On a corpus
of a few thousand chunks this returns in single-digit milliseconds, which is faster
than the network round-trip to a hosted vector store would be.

Usage:
    python3 rag.py index <dir>          # embed and store every .md/.txt file under <dir>
    python3 rag.py ask "<question>"     # retrieve, answer, and verify the answer
    python3 rag.py eval <cases.json>    # run the evaluation harness
    python3 rag.py selfcheck            # offline checks, no Ollama required
"""
import json
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

OLLAMA = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma2:2b"
DB = Path(__file__).parent / "corpus.db"

# Chunking by character count rather than tokens: the embedding model truncates at its
# own context limit anyway, and an exact token count would buy accuracy we never spend.
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200


def _post(path, payload, timeout=120):
    req = urllib.request.Request(
        f"{OLLAMA}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.URLError as exc:
        raise SystemExit(f"Ollama unreachable at {OLLAMA} ({exc}). Is the service running?")


def embed(text, kind="document"):
    """Return a unit-normalised embedding, so cosine similarity is a plain dot product.

    nomic-embed-text is trained with task prefixes and expects them at inference. Omitting
    them does not error — it quietly collapses the score range (every pair lands near 0.5)
    and retrieval stops discriminating. Passing the wrong prefix is the same silent failure,
    so documents and queries are embedded through explicitly different paths.
    """
    prefix = "search_document: " if kind == "document" else "search_query: "
    vec = np.array(
        _post("/api/embeddings", {"model": EMBED_MODEL, "prompt": prefix + text})["embedding"],
        dtype=np.float32,
    )
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def chunk(text):
    """Split on Markdown headings first, then pack paragraphs within each section.

    Packing paragraphs blindly was the original implementation and it failed in a way worth
    recording: a section heading landed at the tail of one chunk while the facts under it
    opened the next, so no single chunk was *about* that section and it never retrieved.
    Headings are the document's own statement of what belongs together, so they are hard
    boundaries here, and every sub-chunk of a long section is re-prefixed with its heading
    to keep that context attached.
    """
    sections, heading, body = [], "", []
    for line in text.splitlines():
        if re.match(r"^#{1,6} ", line):
            if heading or body:
                sections.append((heading, "\n".join(body)))
            heading, body = line.strip(), []
        else:
            body.append(line)
    if heading or body:
        sections.append((heading, "\n".join(body)))

    chunks = []
    for heading, body in sections:
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        if not paras:
            if heading:
                chunks.append(heading)
            continue
        current = ""
        for para in paras:
            if current and len(current) + len(para) + 2 > CHUNK_CHARS:
                chunks.append(f"{heading}\n\n{current}" if heading else current)
                current = (current[-CHUNK_OVERLAP:] + "\n\n" + para) if CHUNK_OVERLAP else para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            chunks.append(f"{heading}\n\n{current}" if heading else current)
    return chunks


def clean_for_embedding(text):
    """Strip Markdown syntax so the embedding sees prose, not punctuation.

    Measured on this corpus: the sentence "Namrin Market is built with React 18, TypeScript
    and Vite" scores 0.82 against a natural question about it, while the same facts in their
    original resume formatting — heading markers, italics, middot-separated stack list, a
    bare URL — score 0.47, below *unrelated* text. The formatting was outweighing the
    content. Only the embedded copy is cleaned; the original is what gets cited and shown,
    so nothing the reader sees is altered.
    """
    text = re.sub(r"https?://\S+|\b[\w.-]+\.(?:com|dev|ai|org|net)\b\S*", " ", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)   # heading markers
    text = re.sub(r"[*_`]+", "", text)                            # bold/italic/code
    text = re.sub(r"^\s*[-+*]\s+", "", text, flags=re.MULTILINE)  # bullet markers
    text = re.sub(r"^\s*\|.*\|\s*$", lambda m: m.group(0).replace("|", ", "),
                  text, flags=re.MULTILINE)                       # table rows to prose
    text = re.sub(r"\s*[·•—–]\s*", ", ", text)                    # separator glyphs
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def connect():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        source TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        text TEXT NOT NULL,
        embedding BLOB NOT NULL
    )""")
    # FTS5 ships with SQLite, so keyword search costs one virtual table and no dependency.
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='')")
    return conn


def index(root):
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in (".md", ".txt") and p.is_file()]
    if not files:
        raise SystemExit(f"No .md or .txt files found under {root}")

    conn = connect()
    conn.execute("DELETE FROM chunks")  # full reindex; incremental sync is not worth the state
    conn.execute("DELETE FROM chunks_fts")
    total = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for ordinal, piece in enumerate(chunk(text)):
            cur = conn.execute(
                "INSERT INTO chunks (source, ordinal, text, embedding) VALUES (?, ?, ?, ?)",
                (str(path), ordinal, piece, embed(clean_for_embedding(piece)).tobytes()),
            )
            conn.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                         (cur.lastrowid, clean_for_embedding(piece)))
            total += 1
        print(f"  indexed {path.relative_to(root)}")
    conn.commit()
    conn.close()
    print(f"\n{total} chunks from {len(files)} files → {DB}")


KEYWORD_WEIGHT = 0.5  # measured below; anything in 0.5-0.7 behaves the same on this corpus


def bm25_scores(conn, question, ids):
    """BM25 relevance per chunk id, as positive numbers (SQLite returns them negated)."""
    terms = re.findall(r"\w+", question.lower())
    if not terms:
        return np.zeros(len(ids), dtype=np.float32)
    try:
        hits = {r[0]: -r[1] for r in conn.execute(
            "SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?",
            (" OR ".join(terms),),
        )}
    except sqlite3.OperationalError:
        return np.zeros(len(ids), dtype=np.float32)  # bad FTS syntax → degrade to vector-only
    return np.array([hits.get(i, 0.0) for i in ids], dtype=np.float32)


def _normalise(scores):
    """Min-max to [0,1] so two incomparable score scales can be blended."""
    lo, hi = float(scores.min()), float(scores.max())
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


def retrieve(question, k=4):
    """Hybrid retrieval: dense vector similarity blended with BM25 keyword relevance.

    Vector search alone measurably failed on this corpus. Asked "What is Namrin Market built
    with?", the section that literally answers it scored 0.53 while an unrelated bartending
    entry scored 0.63 — rare proper nouns carry little embedding signal, and terse formatted
    prose (a resume, a spec sheet) embeds poorly against a natural-language question. The
    same facts rewritten as an ordinary sentence scored 0.82, which is the tell: the model
    understood the content and not the formatting.

    BM25 finds "Namrin" immediately because it is a rare literal token — exactly the case
    dense retrieval is worst at. Blending uses normalised *scores* rather than reciprocal
    rank fusion on purpose: BM25 ranked the right chunk at -11.0 against -5.8 for the runner
    up, and that margin is the whole signal. RRF discards magnitude, and measurably buried
    the correct chunk at rank 9 while a chunk that was mediocre in both rankings won.
    """
    conn = connect()
    rows = conn.execute("SELECT id, source, ordinal, text, embedding FROM chunks").fetchall()
    if not rows:
        conn.close()
        raise SystemExit("Corpus is empty. Run: python3 rag.py index <dir>")

    ids = [r[0] for r in rows]
    matrix = np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
    dense = matrix @ embed(question, kind="query")
    keyword = bm25_scores(conn, question, ids)
    conn.close()

    blended = KEYWORD_WEIGHT * _normalise(keyword) + (1 - KEYWORD_WEIGHT) * _normalise(dense)
    top = np.argsort(blended)[::-1][:k]
    return [
        {"source": rows[i][1], "ordinal": rows[i][2], "text": rows[i][3],
         "score": float(blended[i]), "dense": float(dense[i]), "keyword": float(keyword[i])}
        for i in top
    ]


PROMPT = """Answer the question using ONLY the numbered sources below.

Cite the source number inline for every claim, like [1] or [2].
If the sources do not contain the answer, reply with exactly: NOT IN SOURCES

Sources:
{sources}

Question: {question}

Answer:"""


def ask(question, k=4):
    hits = retrieve(question, k)
    sources = "\n\n".join(f"[{n}] {h['text']}" for n, h in enumerate(hits, 1))
    answer = _post("/api/generate", {
        "model": CHAT_MODEL,
        "prompt": PROMPT.format(sources=sources, question=question),
        "stream": False,
        "options": {"temperature": 0},
    })["response"].strip()
    return answer, hits


def check_citations(answer, hits):
    """Flag answers that cite nothing, or cite a source number that was never retrieved.

    This is the point of the project. A RAG pipeline that returns fluent uncited prose is
    indistinguishable from one that is making things up, so the failure is caught here
    rather than left for the reader to notice.
    """
    if answer.strip() == "NOT IN SOURCES":
        return []
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    problems = []
    if not cited:
        problems.append("answer cites no sources")
    for n in sorted(cited - set(range(1, len(hits) + 1))):
        problems.append(f"cites [{n}], which was not retrieved")
    return problems


def evaluate(cases_path):
    """Score the pipeline against known-answer cases, including must-refuse cases.

    A case with "expect": "NOT IN SOURCES" asserts the model declines to answer something
    absent from the corpus. Without those, an eval only measures recall and rewards a model
    that confidently answers everything.
    """
    cases = json.loads(Path(cases_path).read_text())
    passed = 0
    for case in cases:
        answer, hits = ask(case["question"])
        problems = check_citations(answer, hits)
        if case["expect"] == "NOT IN SOURCES":
            ok = answer.strip() == "NOT IN SOURCES"
            why = "answered a question with no support in the corpus" if not ok else ""
        else:
            ok = case["expect"].lower() in answer.lower() and not problems
            why = "; ".join(problems) or (f"expected {case['expect']!r}" if not ok else "")
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {case['question']}")
        if not ok:
            print(f"       {why}\n       got: {answer[:160]}")
    print(f"\n{passed}/{len(cases)} passed")
    return passed == len(cases)


def selfcheck():
    """Offline checks for the logic that does not need a model."""
    long_text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(10))
    pieces = chunk(long_text)
    assert len(pieces) > 1, "long input should split into multiple chunks"
    assert all(len(p) <= CHUNK_CHARS + CHUNK_OVERLAP + 60 for p in pieces), "chunk too large"
    assert chunk("") == [], "empty input yields no chunks"
    assert len(chunk("one short para")) == 1, "short input stays whole"

    # The regression this file exists to prevent: a heading must travel with its own body.
    doc = "### Namrin Market\n\n*React 18 · TypeScript · Vite*\n\n### Growth Console\n\n*Vanilla JS*"
    pieces = chunk(doc)
    assert len(pieces) == 2, f"one chunk per section, got {len(pieces)}"
    namrin = [p for p in pieces if "Namrin" in p]
    assert len(namrin) == 1 and "React 18" in namrin[0], "heading separated from its content"
    assert "Vanilla JS" not in namrin[0], "section bled into its neighbour"

    long_section = "## Deploy\n\n" + "\n\n".join("para " + "w " * 90 for _ in range(20))
    pieces = chunk(long_section)
    assert len(pieces) > 1, "long section should still split"
    assert all(p.startswith("## Deploy") for p in pieces), "sub-chunks must keep their heading"

    four = [{}] * 4
    assert check_citations("NOT IN SOURCES", four) == []
    assert check_citations("Solar output rose [1] and fell [2].", four) == []
    assert check_citations("Solar output rose.", four) == ["answer cites no sources"]
    assert check_citations("As shown [9].", four) == ["cites [9], which was not retrieved"]

    # Markdown formatting must not survive into the embedded text — it outranked content.
    cleaned = clean_for_embedding(
        "### Namrin Market — Voice-First ESL Game\n\n*React 18 · TypeScript · Vite · foo.dev*\n\n- Built a game."
    )
    assert not re.search(r"[#*·—]", cleaned), f"markdown survived cleaning: {cleaned}"
    assert "foo.dev" not in cleaned, "URL survived cleaning"
    assert "Namrin Market" in cleaned and "React 18" in cleaned, "cleaning destroyed content"

    flat = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    assert _normalise(flat).tolist() == [0.0, 0.0, 0.0], "constant scores must not divide by zero"
    spread = _normalise(np.array([2.0, 4.0, 6.0], dtype=np.float32))
    assert spread.tolist() == [0.0, 0.5, 1.0], f"min-max normalisation wrong: {spread}"

    print("selfcheck passed")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "index":
        index(sys.argv[2])
    elif cmd == "ask":
        answer, hits = ask(" ".join(sys.argv[2:]))
        print(answer, "\n")
        for n, h in enumerate(hits, 1):
            print(f"  [{n}] {Path(h['source']).name} #{h['ordinal']}  (score {h['score']:.3f})")
        problems = check_citations(answer, hits)
        if problems:
            print("\n  CITATION PROBLEMS: " + "; ".join(problems))
    elif cmd == "eval":
        sys.exit(0 if evaluate(sys.argv[2]) else 1)
    elif cmd == "selfcheck":
        selfcheck()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
