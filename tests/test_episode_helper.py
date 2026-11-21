"""
Tests for shared Episode Formatting & Presentation Helper.
==========================================================
Verifies MM:SS.SS formatting, gap explanations, and analytical facts.
"""

import pytest
from dashboard.views.episode_helper import (
    format_timestamp_mmss,
    get_episode_gap_summary,
)


def test_format_timestamp_mmss():
    assert format_timestamp_mmss(0.0) == "00:00.00"
    assert format_timestamp_mmss(5.25) == "00:05.25"
    assert format_timestamp_mmss(59.99) == "00:59.99"
    assert format_timestamp_mmss(60.0) == "01:00.00"
    assert format_timestamp_mmss(72.8) == "01:12.80"
    assert format_timestamp_mmss(79.6) == "01:19.60"
    assert format_timestamp_mmss(125.45) == "02:05.45"
    # Edge cases
    assert format_timestamp_mmss(None) == "00:00.00"
    assert format_timestamp_mmss(-10.0) == "00:00.00"


def test_get_episode_gap_summary_no_gaps():
    ep = {
        "episode_index": 1,
        "duration_sec": 3.4,
        "has_gaps": False,
        "gaps": [],
    }
    note, color = get_episode_gap_summary(ep)
    assert note == "No missing sampled observations recorded."
    assert color == "#059669"


def test_get_episode_gap_summary_single_gap():
    ep = {
        "episode_index": 2,
        "duration_sec": 6.8,
        "has_gaps": True,
        "gaps": [
            {
                "missing_sample_count": 1,
                "gap_duration_sec": 0.4,
            }
        ],
    }
    note, color = get_episode_gap_summary(ep)
    assert note == "One tracking sample is missing. The video is available for review."
    assert color == "#475569"
    # Verify no continuous presence assertion
    assert "continuous" not in note.lower()


def test_get_episode_gap_summary_multi_gap():
    ep = {
        "episode_index": 3,
        "duration_sec": 12.0,
        "has_gaps": True,
        "gaps": [
            {"missing_sample_count": 1},
            {"missing_sample_count": 2},
        ],
    }
    note, color = get_episode_gap_summary(ep)
    assert note == "3 tracking samples are missing. The video is available for review."
    assert color == "#475569"
