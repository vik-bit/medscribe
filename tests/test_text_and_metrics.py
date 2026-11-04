import math

import pytest

from medscribe.asr.metrics import cluster_bootstrap, paired_cluster_bootstrap, score_utterance
from medscribe.eval.note_metrics import (
    claim_grounded,
    highlight_recall,
    split_reference_claims,
    token_f1,
    ungrounded_rate,
)
from medscribe.stats import bootstrap_ci, paired_bootstrap_diff
from medscribe.text import normalise_tokens, numbers_in


def test_shorthand_and_durations_expand():
    assert normalise_tokens("3/7 hx of diarrhoea") == ["3", "days", "history", "diarrhoea"]
    assert "past" in normalise_tokens("PMH: asthma")
    assert normalise_tokens("Nil fever") == ["no", "fever"]  # negation kept


def test_number_words_map_to_digits():
    assert numbers_in("about five days, two puffs") == {"5", "2"}
    assert numbers_in("10mg once daily") >= {"1"}


def test_token_f1_identity_and_disjoint():
    assert token_f1("cough five days", "5/7 cough")[2] > 0.5
    assert token_f1("headache", "cough") == (0.0, 0.0, 0.0)


def test_highlight_recall():
    assert (
        highlight_recall(
            "Viral chest infection. Cough 5 days.", ["viral chest infection", "salbutamol"]
        )
        == 0.5
    )
    assert math.isnan(highlight_recall("x", []))


def test_claim_grounding_catches_invented_numbers(transcript):
    turns = [t.text for t in transcript[0].turns]
    assert claim_grounded("Cough for five days", turns)
    assert not claim_grounded("Cough for nine days", turns)
    assert not claim_grounded("Takes amoxicillin 500mg", turns)
    assert ungrounded_rate(["Cough for five days", "Has diabetes"], turns) == 0.5


def test_reference_claim_split():
    assert len(split_reference_claims("5/7 cough. Nil fever.\nPlan: rest")) == 3


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4])
    assert lo <= m <= hi and m == pytest.approx(0.25)
    d, lo, hi, p = paired_bootstrap_diff([1, 1, 1], [0, 0, 0])
    assert d == 1 and p == 0


def test_wer_scoring_and_cluster_ci():
    s = score_utterance("c1", "the cat sat", "the cat sat down")
    assert (s.word_errors, s.ref_words) == (1, 3)
    assert score_utterance("c1", "", "x") is None
    assert score_utterance("c1", "a b", "").word_errors == 2
    scores = [score_utterance(f"c{i % 3}", "a b c d", "a b c x") for i in range(9)]
    r = cluster_bootstrap(scores, n=500)
    assert r["wer"] == pytest.approx(0.25) and r["wer_lo"] <= 0.25 <= r["wer_hi"]
    better = [score_utterance(s.cid, "a b c d", "a b c d") for s in scores]
    diff = paired_cluster_bootstrap(better, scores, n=500)
    assert diff["wer_diff"] == pytest.approx(-0.25)
