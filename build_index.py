import logging

from src.indexing import run_indexing
from src.ingestion import run_ingestion
from src.preprocessing import run_preprocessing


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    papers = run_ingestion()
    chunks = run_preprocessing(papers)
    count = run_indexing(chunks)
    print(f"Indexed {len(papers)} papers and {count} chunks.")


if __name__ == "__main__":
    main()
