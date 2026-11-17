# Between the Lines

Find books by meaning, not keywords. This repository contains a semantic book
recommendation engine for Duke AIPI 510. Book descriptions are embedded offline
with `sentence-transformers/all-MiniLM-L6-v2` and stored in a persistent Chroma
collection. The deployed app will query Chroma; normalized NumPy retrieval is
retained as an exact, independently tested reference implementation.

## Current status

Phases 1-5 are implemented; only creation of the public Streamlit app remains.
The offline pipeline cleans and deduplicates a
streamed source, performs deterministic seeded sampling, creates normalized
embeddings, writes aligned CSV/NumPy/JSON artifacts, and batch-ingests them into
an embedded Chroma cosine collection. The Streamlit app loads the validated
collection and CPU embedding model once, accepts free-text prompts, and displays
5 or 10 ranked results with accurately labeled cosine similarities. Tests use a
fake embedder and temporary persistent Chroma client, so routine tests do not
download a dataset or transformer.

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

### Usage and temporary-demo status

The dataset card cites the UCSD Goodreads research but does **not** declare a
license or an explicit redistribution grant. This project uses it only for a
short-lived, non-commercial course demonstration at the repository owner's
direction. That does not establish redistribution rights. The public app and
generated deployment index should be removed after the one-week demonstration,
and the source terms should be confirmed before any longer-lived publication.

To reduce exposure, `books.csv` and `embeddings.npy` remain ignored and are not
published. The deployable Chroma store necessarily contains the selected text
needed to return results, so making the app/repository public still exposes a
derived data artifact; it should not be described as zero redistribution.

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

## Run the application

The app expects an index beneath `data/` by default:

```bash
streamlit run app.py
```

To use a development or externally restored index, set its directory:

```bash
BOOK_INDEX_DIR=data/dev streamlit run app.py
```

The model and Chroma collection are cached per app process. The app does not
re-embed the corpus at startup. Missing or malformed artifacts and failed
queries produce visitor-friendly messages rather than exposed tracebacks.

## Chroma design

The pipeline uses `chromadb.PersistentClient(path="data/chroma")` and explicit
precomputed MiniLM embeddings. Each Chroma record contains the stable internal
book ID, exact embedded description, and title metadata. The collection stores
the model ID, dimension, row count, and corpus fingerprint and is checked after
ingestion. Chroma uses cosine distance for nearest-neighbor queries.

For deployment, the repository includes only `data/manifest.json` and a packaged
`data/chroma/`. The 141 MiB SQLite file is reproducibly gzip-compressed below
GitHub's 100 MiB per-file limit, checked by SHA-256, and restored into `/tmp` at
app startup; the smaller HNSW files remain alongside it. Create the package with
`python scripts/package_chroma.py`. The raw corpus export (`data/books.csv`),
duplicate NumPy matrix (`data/embeddings.npy`), and uncompressed SQLite database
remain ignored. The app validates the restored Chroma row count, model,
dimension, collection name, and corpus fingerprint before serving queries. This
avoids rebuilding 20,000 embeddings on every Streamlit restart.

## Manual semantic smoke evaluation

The final 20,000-book corpus returned these top-three results during the local
Phase 5 smoke test:

| Query | Top results |
| --- | --- |
| atmospheric mystery in a secretive small town | *The Wickerman*; *Haunted Missouri*; *The Trials* |
| dangerous journey through space and distant worlds | *Starrigger*; *What Is Your Dangerous Idea?*; *Outside of Space* |
| moving story about family relationships and loss | *Realer Than Real*; *Great Topics of the World: Essays*; *Missing Hailey* |

This confirms that real rows flow through MiniLM, Chroma, and the UI; it is not
an accuracy claim. Several results are only loosely relevant, which is useful
evidence of the limitations of description-only nearest-neighbor retrieval. A
more meaningful evaluation should compare this system with a TF-IDF baseline
using a small manually judged query set.

## Deploy on Streamlit Community Cloud

1. Push `main` to the GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the
   GitHub account that can access the repository.
3. Choose **Create app**, select `vagizD/book_retrieval`, branch `main`, and
   entrypoint `app.py`.
4. Keep the default Python runtime from `runtime.txt`; no secrets are required.
5. Deploy, then verify that the corpus caption appears and a query returns five
   real rows rather than an index error.

After the course demonstration, delete the Community Cloud app and remove the
generated `data/chroma/` and `data/manifest.json` files from the public branch.
Deleting only the current files does not erase Git history; if permanent removal
is required, rewrite the affected history or replace the public repository.

## Remaining deployment steps

1. Push the verified Phase 5 commit and deployment artifacts.
2. Create the Streamlit Community Cloud app and verify its public URL.
