from medscribe.data.primock import make_splits, parse_textgrid
from tests.conftest import FIXTURE


def test_parse_textgrid_handles_escaped_quotes():
    tg = 'xmin = 0 \n xmax = 1.5 \n text = "he said ""ow"" <UNSURE>loudly</UNSURE>" \n'
    (iv,) = parse_textgrid(tg)
    assert iv.end == 1.5 and iv.text == 'he said "ow" <UNSURE>loudly</UNSURE>'


def test_merge_orders_turns_by_time_and_keeps_roles(consultation):
    roles = [u.speaker for u in consultation.utterances]
    assert roles == ["DOCTOR", "PATIENT"] * 4 + ["DOCTOR"]
    assert [u.uid for u in consultation.utterances] == list(range(9))
    assert consultation.highlights[1] == "viral chest infection"


def test_diarized_hides_roles_and_returns_gold_map(consultation):
    utts, gold = consultation.diarized()
    assert {u.speaker for u in utts} == {"SPEAKER_A", "SPEAKER_B"}
    assert sorted(gold.values()) == ["DOCTOR", "PATIENT"]
    for orig, anon in zip(consultation.utterances, utts, strict=True):
        assert gold[anon.speaker] == orig.speaker
    assert consultation.diarized() == (utts, gold)  # deterministic


def test_splits_are_disjoint_deterministic_and_complete():
    ids = [f"day{d}_consultation{i:02d}" for d in range(1, 6) for i in range(1, 13)][:57]
    s = make_splits(ids)
    assert s == make_splits(list(reversed(ids)))
    assert sum(len(v) for v in s.values()) == 57
    assert not (set(s["train"]) & set(s["test"])) and not (set(s["dev"]) & set(s["test"]))


def test_fixture_is_marked_synthetic():
    assert "Not PriMock57" in (FIXTURE / "README.md").read_text()
