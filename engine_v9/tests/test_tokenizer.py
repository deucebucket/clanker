from engine_v9.tokenizer import tokenize


def test_basic_split():
    tokens = tokenize("I am happy")
    assert tokens == ["i", "am", "happy"]


def test_compound_bond():
    tokens = tokenize("I got laid off from work")
    assert "laidoff" in tokens
    assert "laid" not in tokens


def test_trigram_compound():
    tokens = tokenize("I got laid off yesterday")
    assert "laidoff" in tokens


def test_passed_away():
    tokens = tokenize("she passed away last night")
    assert "passedaway" in tokens


def test_punctuation_stripped():
    tokens = tokenize("I'm happy!")
    assert "happy" in tokens


def test_no_one_becomes_nobody():
    tokens = tokenize("no one cares")
    assert "nobody" in tokens


def test_empty_string():
    tokens = tokenize("")
    assert tokens == []


def test_preserves_contractions():
    tokens = tokenize("I can't believe it")
    assert "cant" in tokens


def test_cancer_free():
    tokens = tokenize("finally cancer free")
    assert "cancerfree" in tokens
