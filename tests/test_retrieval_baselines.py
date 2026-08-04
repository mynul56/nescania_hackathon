from src.retrieval_baselines import compute_token_f1, compute_rouge_l, normalize_text


def test_normalize_text():
    assert normalize_text("  রোগীর   ব্যথা  ") == "রোগীর ব্যথা"
    assert normalize_text(None) == ""


def test_token_f1_exact():
    res = compute_token_f1("আমার জ্বর এসেছে", "আমার জ্বর এসেছে")
    assert res["f1"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0


def test_rouge_l_partial():
    res = compute_rouge_l("আমার বুকে ব্যথা এবং জ্বর", "আমার বুকে ব্যথা")
    assert res["recall"] == 1.0
    assert res["precision"] < 1.0
