from types import SimpleNamespace
from unittest.mock import Mock

from dashboard import run_adapter
from dashboard.theme import render_metric_card_html


def test_large_video_registered_without_embedding(monkeypatch, tmp_path):
    video = tmp_path / "large.webm"
    with video.open("wb") as stream:
        stream.truncate(26 * 1024 * 1024)
    manager = SimpleNamespace(add=Mock(return_value="/media/verified.webm"))
    from streamlit import runtime
    monkeypatch.setattr(runtime, "exists", lambda: True)
    monkeypatch.setattr(runtime, "get_instance", lambda: SimpleNamespace(media_file_mgr=manager))
    component = Mock(return_value=None)
    monkeypatch.setattr(run_adapter, "_synchronized_player_component", component)
    run = SimpleNamespace(artifacts={"tracking_video": str(video)}, bounds=(0, 8, 0, 9), cell_size=(0.5, 0.75), zones={}, zone_colors={}, zone_names={}, candidate_hotspot_cell=None)
    run_adapter.render_synchronized_player(run, {}, key="large-run")
    manager.add.assert_called_once_with(str(video), "video/webm", "synchronized-player-large-run")
    assert component.call_args.kwargs["video_url"] == "/media/verified.webm"
    assert "video_base64" not in component.call_args.kwargs
    assert component.call_args.kwargs["run_geometry"]["cell_size"] == (0.5, 0.75)


def test_missing_metric_renders_without_crashing():
    assert "Unavailable" in render_metric_card_html("Candidate", None)
    assert ">0<" in render_metric_card_html("Count", 0)
