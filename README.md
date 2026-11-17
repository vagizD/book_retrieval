# Between the Lines

Find books by meaning, not keywords. This repository contains a semantic book
recommendation engine for Duke AIPI 510. Book descriptions are embedded offline
with `sentence-transformers/all-MiniLM-L6-v2` and stored in a persistent Chroma
collection. The deployed app will query Chroma; normalized NumPy retrieval is
retained as an exact, independently tested reference implementation.

## Current status

Phases 1-3 are complete. The offline pipeline now cleans and deduplicates a
streamed source, performs deterministic seeded sampling, creates normalized
embeddings, writes aligned CSV/NumPy/JSON artifacts, and batch-ingests them into
an embedded Chroma cosine collection. Tests exercise the complete pipeline with
a fake embedder and temporary persistent Chroma client, so tests do not download a
dataset or transformer. The Streamlit UI is not implemented yet.

## Verified dataset facts

The proposed source is
[`booksouls/goodreads-book-descriptions`](https://huggingface.co/datasets/booksouls/goodreads-book-descriptions).
Its current Hugging Face card/viewer reports:

- one `train` split with 1,021,106 rows;
- Parquet storage (about 630 MB download size);
- exactly two string fields, `title` and `description`;
- English as the declared language, while warning that a small number of rows
  may be non-English;
- no author, stable source book ID, or language-per-row field.

Because author metadata is absent, the pipeline will deduplicate on normalized
title plus normalized description and assign a stable internal `book_id` after
deterministic selection. It will embed the cleaned description verbatim and
record corpus/index alignment in a manifest.

### Redistribution status

The dataset card cites the UCSD Goodreads research but does **not** declare a
license or an explicit redistribution grant. Therefore generated corpus and
embedding artifacts are ignored and must not be committed or publicly hosted
until the applicable source terms and redistribution permission are confirmed.
This is a deployment blocker, not evidence that redistribution is permitted.
If permission cannot be established, a replacement dataset with explicit terms
must be proposed before changing sources.

## Planned repository boundaries

- `src/retrieval.py`: exact NumPy scoring reference implementation
- `src/preprocessing.py`: cleaning, deduplication, and deterministic sampling
- `src/embeddings.py`: embedding, artifact validation, and Chroma ingestion
- `scripts/build_index.py`: reproducible command-line indexing pipeline
- `app.py`: cached Streamlit UI over the persistent Chroma collection

The index contract fixes the model to
`sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), defaults to 20,000
cleaned books and seed 42, and records the dataset revision, row count, model,
dimension, seed, and corpus SHA-256 fingerprint.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest
python scripts/build_index.py --help
```

### Build a small development index

This scans only 5,000 source rows and is intended to verify the environment,
not to create the production corpus:

```bash
python scripts/build_index.py \
  --max-books 100 \
  --source-limit 5000 \
  --output-dir data/dev
```

The first run downloads the source shard and MiniLM model. CPU is the default.
Generated files are `books.csv`, `embeddings.npy`, `manifest.json`, and the
`chroma/` persistence directory beneath the selected output directory.

### Build the complete index

```bash
python scripts/build_index.py --max-books 20000 --seed 42
```

The complete build scans the streamed source rather than taking its first
20,000 rows. Selection is determined by a seeded SHA-256 priority, and output
rows are stored in a stable order. The manifest records the immutable Hugging
Face revision resolved at build time, model, dimensions, row count, seed,
collection name, and corpus fingerprint. Re-running the command recreates the
named Chroma collection to prevent stale rows.

Do not use `--source-limit` for a production build because it intentionally
limits selection to an ordered prefix of the source.

## Chroma design

The pipeline uses `chromadb.PersistentClient(path="data/chroma")` and explicit
precomputed MiniLM embeddings. Each Chroma record contains the stable internal
book ID, exact embedded description, and title metadata. The collection stores
the model ID, dimension, row count, and corpus fingerprint and is checked after
ingestion. Chroma uses cosine distance for nearest-neighbor queries.

The Chroma directory and flat artifacts are currently ignored because the
dataset's redistribution permission is unresolved. They must not be pushed to
the public repository until that blocker is resolved.

## Roadmap

1. Integrate the validated Chroma collection into Streamlit.
2. Resolve redistribution rights, choose a concrete artifact host/commit
   strategy, and document Streamlit Community Cloud deployment.
