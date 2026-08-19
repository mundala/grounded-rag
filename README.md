# grounded-rag

Retrieval-augmented generation over a local document corpus, built to be checkable rather than
impressive. Runs entirely on local inference — no API keys, no data leaving the machine.

```bash
python3 rag.py index docs/           # embed and store a corpus
python3 rag.py ask "your question"   # retrieve, answer, verify citations
python3 rag.py eval cases.json       # score the pipeline against known answers
python3 rag.py selfcheck             # offline logic checks, no model required
```

Requires [Ollama](https://ollama.com) with `nomic-embed-text` and a chat model
(`gemma2:2b` by default). Python dependency: numpy.

---

## What it does differently

Most RAG demos answer confidently and leave you to notice when they are wrong. Two things here
exist to stop that:

**Answers must cite, and citations are checked.** The model is instructed to cite `[n]` for every
claim and to reply `NOT IN SOURCES` when the retrieved context does not contain the answer.
`check_citations()` then verifies the answer cited *something* and that every number it cited was
actually retrieved. An uncited answer is flagged, because fluent uncited prose is indistinguishable
from a fabrication.

**The eval harness includes must-refuse cases.** Half the cases in `cases.json` have no support in
the corpus and assert the pipeline declines to answer. Without those, an eval only measures recall
and quietly rewards a model that answers everything with confidence.

## Design choices

**No vector database.** SQLite holds chunks and embeddings as float32 blobs; retrieval is a numpy
dot product over the table. At a few thousand chunks this returns in single-digit milliseconds —
faster than the network round-trip to a hosted vector store. A dependency that saves no time is
just a dependency.

**No framework.** The whole pipeline is one file of straightforward Python. Every failure below was
found by reading the actual scores, which is harder to do through an abstraction layer.

## Three retrieval bugs, and how they were caught

Each of these was found by the eval harness failing, not by inspection. They are recorded because
the debugging is the interesting part.

### 1. Missing task prefixes

`nomic-embed-text` is trained with `search_document:` / `search_query:` prefixes and expects them at
inference. Omitting them does not raise — it silently collapses the score range so every pair lands
near 0.5 and retrieval stops discriminating. Documents and queries now embed through explicitly
different paths.

### 2. Markdown formatting outweighing content

Retrieval still failed. Measuring directly against the question *"What is Namrin Market built
with?"*:

| Text embedded | Score |
|---|---|
| `Namrin Market is built with React 18, TypeScript and Vite.` | **0.82** |
| An unrelated entry about solar installation | 0.54 |
| An unrelated entry about bartending | 0.51 |
| The real resume section, with `###`, `*`, `·` separators and a URL | **0.47** |

The section that literally answered the question scored *below unrelated text*. Heading markers,
italics, middot-separated stack lists and bare URLs dominated the embedding. `clean_for_embedding()`
now strips Markdown to prose before embedding. The original text is what gets stored, cited and
shown — only the embedded copy is cleaned, so nothing the reader sees is altered.

### 3. Chunking that separated headings from their content

Naive paragraph packing put the `### Namrin Market` heading at the tail of one chunk and the facts
under it at the head of the next, so no single chunk was *about* that section. Chunking now splits
on Markdown headings first and re-prefixes every sub-chunk of a long section with its heading.
`selfcheck` asserts this specific regression.

### 4. Rank fusion discarding the signal that mattered

Even with all three fixed, dense retrieval ranked the correct chunk ~80th of 150 for that query —
rare proper nouns carry little embedding signal. BM25 (SQLite FTS5, no new dependency) ranked it
**first**, at -11.0 against -5.8 for the runner-up.

Reciprocal rank fusion, the standard approach, still buried it at **rank 9** — because RRF uses only
rank position and throws that decisive margin away, letting a chunk that placed moderately in *both*
rankings beat one that won decisively in *one*. Tuning the RRF constant to fix it required pushing
it to k=2, which is overfitting to a single query.

The fix is score-based blending: min-max normalise both signals and take a weighted sum
(`KEYWORD_WEIGHT = 0.5`). This preserves magnitude. Measured across four queries, the correct chunk
reaches rank 1 for all of them, stable anywhere in the 0.5–0.7 weight range rather than balanced on
a knife's edge.

## Current results

```
6/6 passed
```

Three grounded questions answered correctly with valid citations; three unanswerable questions
correctly refused.

## Known limits

- **Full reindex only.** No incremental sync — re-embedding a small corpus takes under a minute and
  change detection is state worth avoiding until that stops being true.
- **Single-corpus.** One SQLite file, no namespacing or multi-tenancy.
- **Chunk-level citation.** Answers cite a chunk, not a line or sentence within it.
- **`gemma2:2b` is a small model**, chosen so this runs on modest hardware. The retrieval layer is
  model-agnostic; a larger local or hosted model slots in by changing `CHAT_MODEL`.
- **The eval set is small** (6 cases). It is enough to have caught four real bugs, which is the
  point, but it is not a benchmark.
