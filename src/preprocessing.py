import json
import re

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNKS_PATH, CHUNK_OVERLAP, CHUNK_SIZE


def clean_text(text):
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def run_preprocessing(papers):
    """Split each page separately so every chunk retains its source page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for paper in papers:
        with pymupdf.open(paper["pdf_path"]) as document:
            for page_number, page in enumerate(document, start=1):
                text = clean_text(page.get_text(sort=True))
                for index, text_chunk in enumerate(splitter.split_text(text)):
                    chunks.append({
                        "chunk_id": f"{paper['paper_id']}_p{page_number}_c{index}",
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "year": paper["year"],
                        "url": paper["url"],
                        "page": page_number,
                        "text": text_chunk,
                    })
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return chunks
