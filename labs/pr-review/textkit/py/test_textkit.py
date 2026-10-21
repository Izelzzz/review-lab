import unittest

from textkit import mean_word_length, truncate, word_count


class TestTextkit(unittest.TestCase):
    def test_word_count(self):
        self.assertEqual(word_count("hello brave new world"), 4)
        self.assertEqual(word_count(""), 0)

    def test_truncate(self):
        self.assertEqual(truncate("hello world", 8), "hello...")
        self.assertEqual(truncate("short", 10), "short")
        with self.assertRaises(ValueError):
            truncate("abc", 2)

    def test_mean_word_length(self):
        self.assertEqual(mean_word_length("hi there"), 3.5)
        self.assertEqual(mean_word_length("  one   two  "), 3.0)


if __name__ == "__main__":
    unittest.main()
