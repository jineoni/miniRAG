import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import numpy as np
import pymupdf
from streamlit.testing.v1 import AppTest

from src import generation, indexing, ingestion, preprocessing, retrieval


class FixedEmbeddings:
    """Deterministic vectors for testing storage, not semantic quality."""

    def encode(self, texts, **kwargs):
        def vector(text):
            return [1.0, 0.0, 0.0] if "traffic" in text.lower() else [0.0, 1.0, 0.0]

        return np.array(vector(texts) if isinstance(texts, str) else [vector(t) for t in texts])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdf = self.root / "paper.pdf"
        with pymupdf.open() as document:
            document.new_page().insert_text((50, 50), "Traffic signals adapt to demand.")
            document.new_page()  # Empty pages must not produce chunks.
            document.new_page().insert_text((50, 50), "Neural networks classify images.")
            document.save(self.pdf)
        self.paper = {
            "paper_id": "testv1", "title": "Test paper", "year": 2026,
            "url": "https://arxiv.org/abs/testv1", "pdf_path": str(self.pdf),
        }

    def make_chunks(self):
        path = self.root / "processed/chunks.jsonl"
        with patch.object(preprocessing, "CHUNKS_PATH", path):
            chunks = preprocessing.run_preprocessing([self.paper])
        self.assertEqual(chunks, [json.loads(line) for line in path.read_text().splitlines()])
        return chunks

    def test_pdf_pages_and_chunk_boundaries(self):
        chunks = self.make_chunks()
        self.assertEqual([chunk["page"] for chunk in chunks], [1, 3])
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))
        with patch.object(preprocessing, "CHUNKS_PATH", self.root / "long.jsonl"), \
                patch.object(preprocessing, "CHUNK_SIZE", 30), \
                patch.object(preprocessing, "CHUNK_OVERLAP", 10), \
                patch.object(preprocessing, "clean_text", return_value="x" * 95):
            chunks = preprocessing.run_preprocessing([self.paper])
        self.assertTrue(all(0 < len(chunk["text"]) <= 30 for chunk in chunks))
        self.assertGreater(len(chunks), 3)

    def test_persistent_search_and_rebuild(self):
        chunks = self.make_chunks()
        with patch.object(indexing, "DB_PATH", self.root / "db"), \
                patch.object(indexing, "load_embedding_model", return_value=FixedEmbeddings()):
            self.assertEqual(indexing.run_indexing(chunks), 2)
            collection = indexing.open_collection()
            results = retrieval.search_similar_chunks("traffic", FixedEmbeddings(), collection, 5)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["page"], 1)
            self.assertAlmostEqual(results[0]["similarity"], 1)
            self.assertEqual(results[0]["source_id"], 1)
            self.assertEqual(results[0]["url"], self.paper["url"])
            with self.assertRaises(ValueError):
                indexing.run_indexing([])
            self.assertEqual(indexing.open_collection().count(), 2)
            indexing.run_indexing(chunks[:1])
            self.assertEqual(indexing.open_collection().count(), 1)

    def test_ingestion_reuses_pdf_and_merges_metadata(self):
        result = SimpleNamespace(
            get_short_id=lambda: "paper", title="Test paper", authors=[],
            published=datetime(2026, 1, 1),
            summary="Abstract", entry_id=self.paper["url"], pdf_url="https://arxiv.org/pdf/test",
        )
        client = MagicMock()
        client.results.side_effect = lambda search: iter([result])
        path = self.root / "metadata.json"
        with patch.object(ingestion, "RAW_DIR", self.root), \
                patch.object(ingestion, "METADATA_PATH", path), \
                patch.object(ingestion.arxiv, "Client", return_value=client), \
                patch.object(ingestion, "urlopen") as download:
            ingestion.run_ingestion()
            papers = ingestion.run_ingestion()
        download.assert_not_called()
        self.assertEqual(len(papers), 1)
        self.assertEqual(len(json.loads(path.read_text())), 1)

    def test_generation_context_and_invalid_citation(self):
        sources = [{**self.make_chunks()[0], "source_id": 1}]
        with patch.object(generation, "urlopen") as send:
            response = send.return_value.__enter__.return_value
            response.read.return_value = json.dumps({"response": json.dumps({
                "answer": "Controllers adapt to demand.", "source_ids": [1],
            })})
            self.assertIn("[1]", generation.generate_answer("How?", sources))
            request = send.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "qwen2.5:3b")
            self.assertIn("Page: 1", payload["prompt"])
            self.assertIn(sources[0]["text"], payload["prompt"])
            response.read.return_value = json.dumps({"response": json.dumps({
                "answer": "Unsupported.", "source_ids": [1, 9],
            })})
            with self.assertRaisesRegex(ValueError, "unknown source"):
                generation.generate_answer("How?", sources)
            response.read.return_value = json.dumps({"response": json.dumps({
                "answer": "Insufficient evidence.", "source_ids": [],
            })})
            self.assertIn("Insufficient evidence", generation.generate_answer("GDP?", sources))

    def test_ollama_unavailable_has_no_cloud_fallback(self):
        sources = [{**self.make_chunks()[0], "source_id": 1}]
        with patch.object(generation, "urlopen", side_effect=URLError("Offline")) as send:
            with self.assertRaisesRegex(ValueError, "Start local Ollama"):
                generation.generate_answer("How?", sources)
            send.assert_called_once()

    def test_empty_inputs_skip_generation(self):
        with patch.object(generation, "urlopen") as send:
            self.assertIn("Insufficient evidence", generation.generate_answer("How?", []))
            send.assert_not_called()
        with self.assertRaises(ValueError):
            retrieval.search_similar_chunks("  ", None, None)

    def test_streamlit_question_and_sources(self):
        source = {**self.make_chunks()[0], "source_id": 1, "similarity": 0.9}
        collection = MagicMock()
        collection.get.return_value = {"metadatas": [self.paper]}
        collection.count.return_value = 2
        with patch.object(indexing, "open_collection", return_value=collection), \
                patch.object(indexing, "load_embedding_model", return_value=FixedEmbeddings()), \
                patch.object(generation, "answer_question", return_value={
                    "answer": "Signals adapt to demand [1].", "sources": [source],
                }) as answer:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
            app.run(timeout=15)
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("How do signals adapt?").run()
            self.assertFalse(app.exception)
            self.assertIn("[1]", app.chat_message[1].markdown[0].value)
            self.assertEqual(len(app.session_state.history), 1)
            answer.assert_called_once()
            app.slider[0].set_value(3).run()
            self.assertEqual(len(app.chat_message), 2)
            answer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
