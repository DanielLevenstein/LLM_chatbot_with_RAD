import unittest

from rag.llm_client import trim_response


class LlmClientTest(unittest.TestCase):
    def test_trim_response(self):
        self.assertEqual("value", trim_response("<think></think>value"))  # add assertion here


if __name__ == '__main__':
    unittest.main()
