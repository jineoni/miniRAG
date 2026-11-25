# Mini RAG: Research Paper QA Assistant

Mini RAG is a learning-oriented project to implement an end-to-end retrieval-augmented generation pipeline using Python. It collects arXiv papers on reinforcement learning for traffic signal control, retrieves relevant passages, and generates answers with paper and page references.

## Key Features

- Automatic arXiv paper collection (10 results per build) with PDF and JSON metadata storage
- Page-based PDF parsing and overlapping text chunks using LangChain
- Local CPU embeddings with `all-MiniLM-L6-v2` (384 dimensions)
- Persistent ChromaDB storage and cosine similarity search
- Local Ollama answers with numbered sources, paper links, and page references
- Streamlit interface with a paper library, Top-K control, and retrieved passages

## Repository Structure

```text
mini-rag/
├── src/
│   ├── config.py          # Paths, models, and pipeline settings
│   ├── ingestion.py       # arXiv search, PDF download, and metadata
│   ├── preprocessing.py   # Page extraction and text chunking
│   ├── indexing.py        # CPU embeddings and persistent vector storage
│   ├── retrieval.py       # Top-K cosine similarity search
│   └── generation.py      # Grounded answers and source ID validation
├── tests/                 # Local pipeline and boundary tests
├── data/                  # Generated locally; excluded from Git
├── app.py                 # Streamlit question-answering interface
├── build_index.py         # Collection → chunking → indexing
└── requirements.txt
```

## Installation

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com/download), then start its local server with cloud features disabled:

```bash
OLLAMA_NO_CLOUD=1 OLLAMA_MODELS=./data/ollama ollama serve
```

In another terminal, download the local model:

```bash
ollama pull qwen2.5:3b
```

The application uses `qwen2.5:3b` through the fixed local endpoint `127.0.0.1:11434`. It has no paid API integration, API keys, or cloud fallback. If Ollama is unavailable, it reports an error. Local inference has no per-request fee, but uses your computer's memory, processing power, and electricity. The model download is approximately 1.9 GB; initial library, model, and paper downloads require internet access. See the [model library](https://ollama.com/library/qwen2.5:3b) and [Ollama cloud-disable setting](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features).

## Usage

Build the knowledge base:

```bash
python build_index.py
```

Run the interface:

```bash
streamlit run app.py
```

Open [localhost:8501](http://localhost:8501) and ask a question, for example: “How does reinforcement learning adapt traffic signal timing?” Expand the retrieved sources to inspect their text, page numbers, cosine similarity scores, and original paper links.

Defaults in `src/config.py`: 10 search results, 1,000-character chunks, 200-character overlap, and Top-K 5. PDF and metadata files are reused across builds; metadata is merged by paper ID. Each build re-embeds the saved corpus and replaces the collection so obsolete chunks do not remain. Restart Streamlit after rebuilding. Delete `data/` before building if you want a fresh corpus after changing the search topic.

Generated files are stored in `data/raw/`, `data/metadata.json`, `data/processed/chunks.jsonl`, and `data/chroma_db/`. Questions reuse the index without downloading or re-embedding the papers.

## Testing

```bash
python -m unittest discover -s tests -v
```

Local tests use synthetic PDFs, fixed embeddings, and mocked local model responses to check page references, chunk boundaries, persistent retrieval, index replacement, invalid citations, and Ollama connection errors. They do not measure the embedding model's retrieval quality or the LLM's factual accuracy. No paid API is used by tests or the application.

After building a real corpus, manually check factual, conceptual, comparative, and unrelated questions. Verify that cited excerpts support the answer and that unsupported questions receive an insufficient-evidence response.

## Limitations

- Text-based PDFs only; no OCR or specialized table, formula, or image parsing
- Dense retrieval only; no reranking or quantitative evaluation
- Long chunks may be truncated by the embedding model's token limit
- Answers can be incorrect despite valid source IDs; inspect the retrieved passages
- Questions are independent; chat history is displayed but not sent to the model
- Local use only; stop the app before rebuilding the index

## Roadmap

- [x] arXiv ingestion and page-based chunking
- [x] Local embeddings and persistent similarity search
- [x] Source-grounded generation and Streamlit interface
- [ ] Evaluation on questions from the collected papers
- [ ] Hybrid search and reranking
