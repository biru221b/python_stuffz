import unittest
import security

class TestValidatePassword(unittest.TestCase):

    def test_valid_password(self):
        self.assertTrue(security.validate_password("Abcdef1!"))

    def test_too_short(self):
        with self.assertRaises(ValueError) as cm:
            security.validate_password("Ab1!")
        self.assertIn("at least", str(cm.exception))

    def test_missing_uppercase(self):
        with self.assertRaises(ValueError) as cm:
            security.validate_password("abcdef1!")
        self.assertEqual(str(cm.exception), "Password must contain an uppercase letter")

    def test_missing_lowercase(self):
        with self.assertRaises(ValueError) as cm:
            security.validate_password("ABCDEF1!")
        self.assertEqual(str(cm.exception), "Password must contain a lowercase letter")

    def test_missing_digit(self):
        with self.assertRaises(ValueError) as cm:
            security.validate_password("Abcdefg!")
        self.assertEqual(str(cm.exception), "Password must contain a digit")

    def test_missing_special_char(self):
        with self.assertRaises(ValueError) as cm:
            security.validate_password("Abcdefg1")
        self.assertEqual(str(cm.exception), "Password must contain a special character")

    def test_contains_space(self):
        with self.assertRaises(ValueError) as cm:
            security.validate_password("Abcdef1 !")
        self.assertEqual(str(cm.exception), "Password must not contain spaces")

    def test_non_string_input(self):
        with self.assertRaises(ValueError) as cm:
            security.validate_password(12345678)
        self.assertEqual(str(cm.exception), "Password must be a string")

    def test_custom_min_length(self):
        # should fail because min_len=12
        with self.assertRaises(ValueError):
            security.validate_password("Abcdef1!", min_len=12)

        # should pass for 12+
        self.assertTrue(security.validate_password("Abcdefghij1!", min_len=12))


if __name__ == "__main__":
    unittest.main()