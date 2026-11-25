import json
import logging
import shutil
import time
from urllib.error import URLError
from urllib.request import urlopen

import arxiv
import pymupdf

from src.config import MAX_PAPERS, METADATA_PATH, RAW_DIR, SEARCH_QUERY

logger = logging.getLogger(__name__)


def run_ingestion():
    """Download missing PDFs and merge metadata by arXiv ID."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else []
    papers = {paper["paper_id"]: paper for paper in existing}
    client = arxiv.Client(page_size=MAX_PAPERS, delay_seconds=3, num_retries=2)
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=MAX_PAPERS,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    for result in client.results(search):
        paper_id = result.get_short_id()
        filename = paper_id.replace("/", "_") + ".pdf"
        path = RAW_DIR / filename
        try:
            if not path.exists():
                time.sleep(3)
                with urlopen(result.pdf_url, timeout=60) as response, path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            with pymupdf.open(path) as document:
                if not document.is_pdf or not len(document):
                    raise ValueError("Not a readable PDF")
        except (OSError, URLError, RuntimeError, ValueError) as error:
            path.unlink(missing_ok=True)
            papers.pop(paper_id, None)
            logger.warning("Skipping %s: %s", paper_id, error)
            continue
        papers[paper_id] = {
            "paper_id": paper_id,
            "title": " ".join(result.title.split()),
            "authors": [author.name for author in result.authors],
            "year": result.published.year,
            "published_date": result.published.date().isoformat(),
            "abstract": result.summary,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "pdf_path": str(path),
        }
        logger.info("Collected %s", paper_id)
    metadata = list(papers.values())
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
