from engine_v9.pipeline import compute_vadug

CENTER = 128

def test_basic_positive():
    result, trace = compute_vadug("I am happy")
    assert result.v > CENTER

def test_basic_negative():
    result, trace = compute_vadug("I am sad")
    assert result.v < CENTER

def test_negation():
    result, trace = compute_vadug("I am not happy")
    assert result.v < CENTER

def test_strong_negative_event():
    result, trace = compute_vadug("I just got laid off from work")
    assert result.v < 100

def test_yoda_order():
    v1, _ = compute_vadug("I just got laid off from work")
    v2, _ = compute_vadug("Laid off from work I just got")
    assert abs(v1.v - v2.v) <= 15

def test_neutral():
    result, _ = compute_vadug("the weather is cloudy today")
    assert 100 < result.v < 156

def test_trace_has_equation():
    _, trace = compute_vadug("I am happy")
    assert "equation" in trace
    assert "nucleus" in trace["equation"]
    assert "subject" in trace["equation"]

def test_trace_has_tokens():
    _, trace = compute_vadug("I am happy")
    assert "tokens" in trace

def test_empty_string():
    result, _ = compute_vadug("")
    assert result.v == CENTER

def test_returns_vadug_type():
    result, _ = compute_vadug("hello world")
    assert hasattr(result, 'v')
    assert hasattr(result, 'a')
    assert hasattr(result, 'd')
    assert hasattr(result, 'u')
    assert hasattr(result, 'g')
    assert hasattr(result, 'w')
    assert hasattr(result, 'i')
