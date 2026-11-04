from medscribe.asr.evaluate import subset
from medscribe.asr.segments import segments_for
from tests.conftest import CID, FIXTURE


def test_segments_cover_both_channels_and_filter_duration():
    segs = segments_for(FIXTURE, [CID])
    assert {s.channel for s in segs} == {"doctor", "patient"}
    assert len(segs) == 9
    assert all(0.5 <= s.duration <= 30 for s in segs)
    assert (
        segments_for(FIXTURE, [CID], max_s=2.0) and len(segments_for(FIXTURE, [CID], max_s=2.0)) < 9
    )


def test_subset_is_seeded_and_ordered():
    segs = segments_for(FIXTURE, [CID])
    a, b = subset(segs, 4), subset(segs, 4)
    assert a == b and len(a) == 4
    assert subset(segs, 0) == segs
