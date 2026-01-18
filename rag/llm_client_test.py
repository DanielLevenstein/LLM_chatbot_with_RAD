import unittest

from rag.llm_client import trim_response, retrieve, INDEX_PATH, SENTENCE_TRANSFORMER, CHUNKS_PATH

from sentence_transformers import SentenceTransformer

import pickle
import faiss

class LlmClientTest(unittest.TestCase):
    def test_trim_response(self):
        self.assertEqual("value", trim_response("<think></think>value"))  # add assertion here

    def test_retrieve_chunks(self):
        index = faiss.read_index(INDEX_PATH)
        model = SentenceTransformer(SENTENCE_TRANSFORMER)
        # Load chunks from file
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        context = retrieve("Blood Pressure", index, chunks, model)
        print(context)
        self.assertNotEqual(len(context[0]), 1, "Chunks should be more than 1 character")
if __name__ == '__main__':
    unittest.main()
