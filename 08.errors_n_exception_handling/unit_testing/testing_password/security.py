import re

def validate_password(pw, min_len=8):
    """
    Rules:
    - must be a string
    - length >= min_len
    - contains at least 1 uppercase, 1 lowercase, 1 digit, 1 special char
    - no spaces allowed
    Returns: True if valid, otherwise raises ValueError with message
    """
    if not isinstance(pw, str):
        raise ValueError("Password must be a string")

    if " " in pw:
        raise ValueError("Password must not contain spaces")

    if len(pw) < min_len:
        raise ValueError(f"Password must be at least {min_len} characters")

    if not re.search(r"[A-Z]", pw):
        raise ValueError("Password must contain an uppercase letter")

    if not re.search(r"[a-z]", pw):
        raise ValueError("Password must contain a lowercase letter")

    if not re.search(r"\d", pw):
        raise ValueError("Password must contain a digit")

    if not re.search(r"[!@#$%^&*()_\-+=\[\]{};:'\",.<>/?\\|`~]", pw):
        raise ValueError("Password must contain a special character")

    return True
