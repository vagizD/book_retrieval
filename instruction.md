# Codex Implementation Instructions — Semantic Book Recommendation Engine

## 1. Project objective

Build a fully functional, publicly deployable **semantic book recommendation engine** for Duke AIPI 510.

The user enters a free-text description of a book they would like to read. The application returns the **top 5 or 10 books** from a real book dataset, ranked by semantic similarity between the user's query and each book's description.

Deliver a working Python repository and a Streamlit application suitable for **Streamlit Community Cloud** deployment. Do not only provide snippets or a prototype notebook.

### Core principles

- **Own the retrieval logic.** Use NumPy to implement normalized vector scoring and top-k selection; do not delegate retrieval to a vector database or orchestration framework.
- **Use a pretrained embedding model, not a hosted LLM.** Default to `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). Do not train a transformer from scratch.
- **Precompute book embeddings offline.** Never re-embed all books when a Streamlit user submits a query.
- **Keep the solution simple.** Avoid LangChain, LlamaIndex, Pinecone, Chroma, FAISS, a backend web server, authentication, and paid APIs in the MVP.
- **Prioritize readable, type-annotated code.** Use focused functions, Google/PEP 8-style docstrings where useful, clear module boundaries, and minimal premature abstraction.
- **Build incrementally.** Work in small, reviewable steps (ideally around 50 lines of meaningful code per iteration where practical). After each step, describe changes, how to verify them, and what is next. Do not generate the entire repository in a single opaque implementation pass.
- **Be reproducible.** Pin or bound dependencies appropriately, record the dataset/model identity, fix random seeds, and document how to rebuild the index.

## 2. Required functional behavior

A visitor to the app can:

1. Enter a natural-language prompt (for example: "A mysterious atmospheric novel about a small town hiding dark secrets").
2. Select either **5** or **10** results (default: 5).
3. Submit the query and receive ranked book recommendations.
4. See each book's **title**, **description excerpt**, and **similarity score**; show **author** only if the dataset actually provides reliable author metadata.
5. Read a brief explanation of semantic similarity and the dataset/model used.
6. See helpful empty-query and failure messages, rather than a traceback.

Book cover art is **optional** and should not hold up the MVP; do not invent cover links or depend on an unverified image service. The results must be real dataset rows, not LLM-generated book titles.

## 3. Dataset and preparation

### Candidate dataset

Start by investigating the Hugging Face dataset:

`booksouls/goodreads-book-descriptions`

Dataset page: https://huggingface.co/datasets/booksouls/goodreads-book-descriptions

**Do not assume its field names, split structure, license, or availability.** Inspect the actual dataset card/schema and use its real columns. If it is unsuitable, propose a replacement openly and explain why before changing sources. Check redistribution/license conditions before committing a processed subset to a public repository.

### MVP corpus

- Target **20,000** usable English-language books initially, configurable via command-line argument.
- Select books deterministically with a fixed seed; don't merely take the first 20,000 rows if the source order creates obvious bias. For very large or streaming datasets, use a reproducible, memory-conscious sampling strategy.
- Remove rows with absent/blank titles or descriptions.
- Remove duplicate records using normalized title plus author when present, otherwise a clearly documented fallback (e.g. normalized title plus description). Avoid unintentionally removing different books that share a title.
- Exclude extremely short/non-informative descriptions using a documented threshold.
- Preserve only needed metadata, including a stable internal `book_id` and the exact text used for embedding.
- Handle long descriptions consistently: MiniLM's maximum sequence length truncates longer text. For the MVP use an explicitly documented truncation strategy; chunking is out of scope.
- Save a clean, reproducible corpus artifact with rows in a known order.

**Important:** The embedding array and metadata must have a verifiable one-to-one row alignment. Store and verify the row count, model identifier, and an index/corpus fingerprint (or equivalent) to detect stale or mismatched artifacts.

## 4. Embedding and indexing pipeline (offline)

Write a script that:

1. Loads or prepares the cleaned dataset.
2. Loads `sentence-transformers/all-MiniLM-L6-v2`.
3. Encodes each book description in manageable batches.
4. Produces `float32` embeddings with `normalize_embeddings=True`.
5. Saves the embedding matrix as `.npy`, using the exact ordering of the metadata rows.
6. Saves metadata/manifest needed to reproduce and validate the artifact.
7. Reports corpus size, embedding shape, elapsed time, and output sizes.

For 20,000 books the embedding array should have shape `(20000, 384)` if all 20,000 survive cleaning. The number may be smaller when cleaning removes records; no hard-coded row count is allowed.

Make the command rerunnable and easy to use, for example:

```bash
python scripts/build_index.py --max-books 20000 --seed 42
```

Do not require a GPU. Optional acceleration is fine, but CPU must be supported.

Do not start encoding the full 1M+ record dataset during routine development or test execution. Build and validate a tiny sample before running the full indexing job.

## 5. Retrieval engine (online)

Implement a small retrieval module, independent of Streamlit, exposing a documented function or class with an interface equivalent to:

```python
def search_books(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    books: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    ...
```

The mathematical operation is:

- Dataset embeddings: `E` of shape `(N, 384)`.
- Embed the query using **the same model** used to build `E`.
- Normalize both book and query embeddings to unit L2 norm.
- Compute all cosine similarities as `scores = E @ query_embedding`.
- Rank by descending similarity and return exactly `min(k, N)` unique rows.

For efficiency, use `np.argpartition` to find the top-k indices and sort only those indices. Properly handle `k == N`, small datasets, empty/whitespace queries, nonfinite vectors, invalid `k`, mismatched dimensions, and mismatched corpus lengths. Decide and document whether empty corpora produce an empty result or a clear exception, and test that behavior.

Scores are **cosine similarity values, not probabilities or calibrated confidence**. Label them accurately in the UI. Do not turn cosine similarity into a fabricated percentage of confidence.

Use exact exhaustive retrieval for the MVP. There is no need for approximate nearest-neighbor search at a 20,000-book scale.

## 6. Streamlit UI

Create `app.py` with:

- Clear product name, e.g. **Between the Lines**.
- Short subtitle: "Find books by meaning, not keywords."
- Multi-line text entry with a helpful example placeholder.
- 5/10 recommendation selector.
- Search button and loading feedback.
- Ranked result cards/sections containing title, author when valid, description excerpt, and cosine similarity (rounded for display).
- Visible caption stating corpus size, embedding model, and that results use description similarity.
- A brief "How it works" section.
- Helpful messages for missing artifacts or a malformed index. Avoid exposing unnecessary exception details to visitors.

Performance and hosting:

- Use `st.cache_resource` for the embedding model and `st.cache_data` or a suitable cached resource for read-only corpus/index artifacts.
- Load the model and precomputed artifacts once per app process; do not rebuild the index at startup.
- Support CPU-only execution on Streamlit Community Cloud.
- Keep memory use modest; avoid unnecessary copies of the embedding matrix.
- Use only public, non-sensitive configuration. No API keys should be needed.
- No heavy custom CSS or complex frontend framework is required.

## 7. Expected repository layout

```text
book-semantic-search/
├── app.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   └── retrieval.py
├── scripts/
│   └── build_index.py
├── data/
│   ├── books.csv             # generated; commit only if redistribution is allowed
│   ├── embeddings.npy        # generated; handle GitHub artifact-size limits
│   └── manifest.json         # generated
├── tests/
│   ├── test_preprocessing.py
│   └── test_retrieval.py
├── requirements.txt
├── .gitignore
├── README.md
└── instruction.md
```

The exact filenames may be adjusted for justified reasons; preserve the separation between preprocessing, indexing, retrieval, and UI.

### Artifacts and hosting requirements

The deployment must actually be able to obtain the precomputed dataset/embeddings. Do not assume that ignored `data/` files will be available in Streamlit Community Cloud.

Pick and document a concrete **deployment artifact strategy**:

- Prefer committing appropriately sized precomputed artifacts if license permits and GitHub limits are respected; **or**
- Host versioned artifacts in a stable public location and download/cache them on startup with integrity checks; **or**
- Use another reproducible approach that does not perform a full embedding rebuild on every deployment.

When preparing the files, check the total repository size and whether Git LFS or external hosting would be required; avoid relying on unconfigured Git LFS. A 20,000 × 384 float32 embedding array alone is about 29.3 MiB, not including corpus metadata or model weights. The model can be downloaded and cached at runtime, subject to Cloud limits.

## 8. Tests and quality checks

Use `pytest` for deterministic tests with small synthetic embeddings/data. Unit tests should not download a transformer or fetch the large dataset.

Required retrieval tests:

1. An exact-match synthetic vector ranks first.
2. Results are in descending score order.
3. `k=5`, `k=10`, and `k > N` return the correct number of rows.
4. Zero/negative `k` is handled consistently.
5. Blank query is rejected before model inference.
6. Embedding/dataframe length mismatch is caught.
7. Dimension mismatch and nonfinite inputs are handled.
8. Duplicate result rows are not returned due to top-k implementation.

Required pipeline checks:

- Cleaned records have usable title and description.
- Deduplication is deterministic.
- Index row alignment is validated when loading artifacts.
- Manifest agrees with the model and embedding dimensions.

Also add a **small manual semantic evaluation** in the README, comparing a few thematic queries against outputs. Suggested follow-up: implement a TF-IDF + cosine similarity baseline and compare using a small, manually judged query/result set. **TF-IDF is optional for MVP**, not a blocker to deployment. Do not make unsupported claims of recommendation accuracy without evaluation data.

## 9. README and deployment instructions

Write a README that explains:

- Problem statement and retrieval architecture.
- Dataset source, license/usage considerations, selection, cleaning, and limitations.
- Embedding model and why it was selected.
- Reproducible environment setup and exact install commands.
- How to generate a small development index and the complete index.
- How to launch locally: `streamlit run app.py`.
- How to run unit tests: `pytest`.
- Where deployment artifacts are stored and how the app downloads/loads them.
- Step-by-step Streamlit Community Cloud deployment from a GitHub repo using `app.py` as the entrypoint.
- Constraints: semantic similarity is not personal preference modeling, quality depends on descriptions and corpus coverage, and similarity scores are not probabilities.

If deployment requires external actions or authentication you cannot perform, explicitly list them as remaining user steps; do not claim the app is publicly deployed until the actual URL is available and verified.

## 10. Implementation order and Codex workflow

Work in these phases, in order:

**Phase 1 — Scaffold and contract**
- Inspect current repo first; preserve existing work.
- Establish files, basic types, CLI contract, and test configuration.
- Verify the dataset schema/license and report the findings.

**Phase 2 — Retrieval before the real model**
- Implement cosine scoring and top-k using synthetic data.
- Implement deterministic unit tests.
- Run tests and fix failures.

**Phase 3 — Offline data pipeline**
- Implement cleaning/deduplication and reproducible corpus sampling.
- Add a tiny fixture/sample run.
- Implement embedding batch generation, save/load, and manifest checks.
- Run a small end-to-end smoke test before generating 20,000 embeddings.

**Phase 4 — Streamlit integration**
- Implement the UI over precomputed embeddings.
- Verify that a local query produces real dataset rows.
- Check startup and query latency on CPU.

**Phase 5 — Deployment readiness**
- Decide artifact distribution and enforce license/size requirements.
- Add dependency versions, README, `.gitignore`, and any deployment configuration.
- Run tests and a reproducible local smoke test.
- Give the user the exact GitHub/Streamlit steps and any remaining blockers.

At the end of **each iteration**, report:

1. Files changed and what was implemented.
2. Exact test/command run and its outcome; clearly say when not run.
3. Next small implementation step.

Avoid unrelated refactoring, speculative abstractions, generated secrets, unnecessary dependencies, or excessive output.

## 11. Definition of done

The MVP is complete when:

- [ ] A reproducible corpus of approximately 20,000 usable books is prepared.
- [ ] Each book description has a corresponding precomputed 384D normalized embedding.
- [ ] A user query is embedded at runtime and ranked with NumPy cosine similarity.
- [ ] The app returns 5 or 10 real books, ranked correctly, with accurate metadata.
- [ ] The UI starts locally and handles expected edge cases.
- [ ] Automated tests pass.
- [ ] README contains executable build, run, test, and deployment commands.
- [ ] The public deployment artifact strategy is functional and compliant with dataset usage rights.
- [ ] Streamlit Community Cloud can load the artifact and model without recomputing the full corpus.
- [ ] No unverified claim of a live URL or successful deployment is made.

## 12. First instruction to Codex

**Begin with Phase 1 only.** Inspect the repository, summarize existing files, verify the proposed Goodreads dataset's actual schema and redistribution conditions, then create only the minimal scaffold needed for the next phase. Explain what you changed and how it was verified. Do not implement the entire system in the first response.
