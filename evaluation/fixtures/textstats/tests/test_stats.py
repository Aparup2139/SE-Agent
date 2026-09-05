from textstats import char_count, longest_word, word_count


def test_word_count_counts_words():
    assert word_count("the quick brown fox") == 4


def test_word_count_of_empty_text_is_zero():
    assert word_count("") == 0


def test_char_count_includes_spaces_by_default():
    assert char_count("ab cd") == 5


def test_char_count_can_exclude_spaces():
    assert char_count("ab cd", include_spaces=False) == 4


def test_longest_word():
    assert longest_word("a bb ccc") == "ccc"


def test_longest_word_of_empty_text():
    assert longest_word("") == ""
