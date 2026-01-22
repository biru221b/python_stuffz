import unittest
import cap

class TestCap(unittest.TestCase):  #inherits from unittest.TestCase

    def test_one_word(self):  #test method for one word
        text = "python"
        result = cap.cap_text(text)
        self.assertEqual(result, "Python")  #Checks if result matches expected value

    def test_multiple_words(self):  #test method for multiple words
        text = "hello world"
        result = cap.cap_text(text)
        self.assertEqual(result, "Hello world")  #Checks if result matches expected value

if __name__ == '__main__':
    unittest.main()