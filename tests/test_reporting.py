import math

from ecbrag.reporting import bootstrap_ci, mean, paired_bootstrap_diff, pct


def test_bootstrap_ci_brackets_the_mean_and_ignores_missing():
    values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, None, float("nan")]
    lo, hi = bootstrap_ci(values)
    assert lo < 0.7 < hi and 0 <= lo and hi <= 1
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0)
    assert all(math.isnan(x) for x in bootstrap_ci([]))


def test_bootstrap_is_wide_on_a_small_set():
    # 14/20 correct: the 95% interval spans roughly 0.5 to 0.9, not a point estimate
    lo, hi = bootstrap_ci([1] * 14 + [0] * 6)
    assert hi - lo > 0.3


def test_paired_diff():
    agent = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    base = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
    r = paired_bootstrap_diff(agent, base)
    assert r["n"] == 10 and abs(r["diff"] - 0.2) < 1e-9 and r["ci_low"] <= 0.2 <= r["ci_high"]


def test_mean_and_pct_skip_missing_values():
    # pandas turns None into NaN in numeric columns: both must be ignored
    assert mean([1.0, None, float("nan"), 3.0]) == 2.0
    assert pct([1, 2, 3, None], 50) == 2.0
    assert math.isnan(mean([None, float("nan")]))
