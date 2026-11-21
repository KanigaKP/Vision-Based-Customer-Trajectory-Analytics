"""Behavioral regressions from the independent final reviews."""
import json
import subprocess
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parents[1]

def app():
    at = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=30)
    at.session_state["authenticated_user"] = "reviewer"
    at.query_params.update(view="studio", run_id="upload_ca30f3e8", stage="6")
    at.run()
    assert not at.exception
    return at

def test_guest_access_to_studio_is_blocked_by_barrier():
    """Verify that unauthenticated guests navigating directly to studio are stopped by the barrier."""
    at = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=30)
    at.query_params.update(view="studio", run_id="upload_ca30f3e8", stage="6")
    at.run()
    assert not at.exception
    barrier_text = " ".join([m.value for m in at.markdown] + [w.value for w in at.warning])
    assert "Protected Feature Access" in barrier_text or "Authentication Required" in barrier_text

def test_run_switch_with_zone_widget_and_missing_artifacts():
    at = app()
    for suffix in ("upload_site02_jewelry", "0a24bbf3de", "demo_real_summary", "upload_6f969f54", "upload_ca30f3e8"):
        selector = at.selectbox(key="studio_run_selector")
        # Widget options are formatted labels; use the canonical discovered values.
        from dashboard.run_adapter import discover_standardized_runs
        target = next(r.run_id for r in discover_standardized_runs(str(ROOT / "outputs/reports")) if suffix in r.run_id)
        selector.set_value(target).run()
        assert not at.exception, suffix
        assert at.session_state["authoritative_run_id"] == target
        focus = [w for w in at.selectbox if w.label == "Focus Geofence / Zone"]
        if focus:
            assert focus[0].key == f"studio_zone_inspector_{target}"
            focus[0].set_value(focus[0].value).run()
            assert not at.exception
        else:
            assert at.session_state["studio_selected_zone"] is None

def test_every_episode_click_issues_a_new_seek_request():
    at = app()
    rid = at.session_state["authoritative_run_id"]
    for expected in (1, 2, 3):
        at.session_state[f"studio_sample_idx_{rid}"] = 188
        next(b for b in at.button if b.label in ("Watch Episode ▶️", "Inspect episode 🔍")).click().run()
        assert not at.exception
        assert at.session_state[f"studio_seek_request_{rid}"] == expected
        assert at.session_state[f"studio_sample_idx_{rid}"] == 167
        assert at.session_state["studio_active_track"] == "1"

def test_production_seek_gate_repeated_target_and_ordinary_reruns():
    module = ROOT / "dashboard/components/synchronized_player/spatial_player_engine.js"
    code = f"const e=require({json.dumps(str(module))}); const g=e.createSeekGate(); console.log(JSON.stringify([g(0,0),g(167,1),g(188,1),g(167,2),g(167,2),g(167,3)]));"
    result = subprocess.run(["node", "-e", code], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [True, True, False, True, False, True]

@pytest.mark.parametrize("density,zone,expected", [(False,True,"ACCOUNTING MISMATCH"),(None,None,"NOT EVALUATED")])
def test_conservation_badge_does_not_claim_success(monkeypatch,density,zone,expected):
    import dashboard.views.spatial_studio as studio
    original = studio.discover_standardized_runs
    def changed(*args,**kwargs):
        runs = original(*args,**kwargs)
        for run in runs:
            run.density_conservation_verified = density
            run.zone_conservation_verified = zone
        return runs
    monkeypatch.setattr(studio,"discover_standardized_runs",changed)
    at=app()
    assert any(expected in item.value for item in at.markdown)
