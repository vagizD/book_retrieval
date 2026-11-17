# Between the Lines

Find books by meaning, not keywords. This repository will contain a semantic
book recommendation engine for Duke AIPI 510. A query will be embedded with
`sentence-transformers/all-MiniLM-L6-v2` and compared with a precomputed book
description index using normalized NumPy dot products.

## Current status

Phases 1 and 2 are complete. The retrieval module performs exact normalized
cosine scoring with NumPy and uses `np.argpartition` before sorting the selected
top-k rows. Synthetic tests cover ranking, result counts, malformed vectors,
alignment errors, blank queries, and empty corpora without downloading a model.
Preprocessing, indexing, and the Streamlit UI are not implemented yet.

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

- `src/retrieval.py`: online NumPy scoring and top-k selection
- `src/preprocessing.py`: cleaning, deduplication, and deterministic sampling
- `src/embeddings.py`: offline embedding generation and artifact validation
- `scripts/build_index.py`: reproducible command-line indexing pipeline
- `app.py`: cached Streamlit UI over precomputed artifacts

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

The build command currently exposes the final CLI contract but exits with a
clear Phase 3 message if invoked beyond `--help`.

## Roadmap

1. Implement deterministic dataset preparation and a tiny indexing smoke test.
2. Integrate precomputed artifacts into Streamlit.
3. Resolve redistribution rights, choose a concrete artifact host/commit
   strategy, and document Streamlit Community Cloud deployment.
