import importlib
import unittest


class TestBasicImport(unittest.TestCase):
    def test_import_package(self):
        module = importlib.import_module("seq_sampler")
        self.assertTrue(hasattr(module, "SeqSampler"))
        self.assertTrue(hasattr(module, "TemporalGraph"))
        self.assertTrue(hasattr(module, "WindowSeqSampler"))


if __name__ == "__main__":
    unittest.main()
