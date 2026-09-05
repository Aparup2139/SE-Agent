"""Basic statistics over a block of text."""


def word_count(text):
    """Return the number of whitespace-separated words in text."""
    if not text:
        return 0
    return len(text.split())


def char_count(text, include_spaces=True):
    """Return the number of characters in text."""
    if not text:
        return 0
    if include_spaces:
        return len(text)
    return len(text.replace(" ", ""))


def longest_word(text):
    """Return the longest whitespace-separated word in text, or '' if empty."""
    words = text.split() if text else []
    if not words:
        return ""
    return max(words, key=len)
