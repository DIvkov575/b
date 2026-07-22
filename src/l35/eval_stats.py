"""Aggregation across multiple independent eval seeds -- turns a list of
per-seed point estimates (e.g. match_rate from N eval runs with different
starting-noise seeds) into a mean + confidence interval, and compares two
configs' per-seed results with a paired significance test.

Needed because evaluate.py's own footnote documents a measurement pipeline
that once swung match_rate by 20 points on an identical checkpoint/seed due
to a generator-state bug: a single n=200/one-seed point estimate from a
pipeline with that history of silent noise is not trustworthy standing
alone, and every comparison in the original paper (student vs. truncated
teacher, same NFE) reused the SAME 200 ground-truth structures across
configs -- a paired test, not an unpaired two-proportion test, is the
statistically correct comparison for that.
"""
import math

from scipy import stats


def aggregate_across_seeds(values, confidence=0.95):
    """Mean + a t-distribution confidence interval across >=2 independent
    seed runs of the same config. Requires at least 2 seeds -- a CI from a
    single point estimate is undefined, and silently returning a
    zero-width interval would look artificially precise rather than
    failing loudly.
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"aggregate_across_seeds needs >=2 seeds, got {n}")

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std_err = math.sqrt(variance / n)

    if std_err == 0.0:
        # All seeds agreed exactly -- a real (if suspicious) outcome, not
        # an error. Report a zero-width interval explicitly rather than
        # NaN from a t-interval with scale=0.
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n_seeds": n}

    ci_low, ci_high = stats.t.interval(confidence, df=n - 1, loc=mean, scale=std_err)
    return {"mean": mean, "ci_low": ci_low, "ci_high": ci_high, "n_seeds": n}


def paired_sign_test(values_a, values_b):
    """Paired (matched-pairs) sign test comparing two configs' per-seed
    results, correct for the case here: both configs are evaluated against
    the SAME ground-truth structures on each seed, so their per-seed
    results are correlated, not independent samples -- an unpaired
    two-proportion test overstates uncertainty and can call a real,
    consistent effect (one config wins on every seed) non-significant.
    """
    if len(values_a) != len(values_b):
        raise ValueError("paired_sign_test requires equal-length paired seed results")

    n_wins_a = sum(1 for a, b in zip(values_a, values_b) if a > b)
    n_wins_b = sum(1 for a, b in zip(values_a, values_b) if b > a)
    n_ties = len(values_a) - n_wins_a - n_wins_b
    n_decisive = n_wins_a + n_wins_b

    if n_decisive == 0:
        return {"n_wins_a": n_wins_a, "n_wins_b": n_wins_b, "n_ties": n_ties, "p_value": 1.0}

    result = stats.binomtest(n_wins_a, n_decisive, p=0.5)
    return {
        "n_wins_a": n_wins_a,
        "n_wins_b": n_wins_b,
        "n_ties": n_ties,
        "p_value": result.pvalue,
    }
