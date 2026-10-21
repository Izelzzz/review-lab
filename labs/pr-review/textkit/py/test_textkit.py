import unittest

from textkit import has_prefix, truncate, word_count


class TestTextkit(unittest.TestCase):
    def test_word_count(self):
        self.assertEqual(word_count("hello brave new world"), 4)
        self.assertEqual(word_count(""), 0)

    def test_truncate(self):
        self.assertEqual(truncate("hello world", 8), "hello...")
        self.assertEqual(truncate("short", 10), "short")
        with self.assertRaises(ValueError):
            truncate("abc", 2)

    def test_has_prefix(self):
        self.assertTrue(has_prefix("RE: meeting", ["RE:", "FW:"]))
        self.assertFalse(has_prefix("meeting", ["RE:", "FW:"]))
        self.assertFalse(has_prefix("meeting", []))
        self.assertTrue(has_prefix("", [""]))


if __name__ == "__main__":
    unittest.main()
