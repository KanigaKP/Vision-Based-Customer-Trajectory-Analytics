import copy
import pytest
from src.spatial.scene_profile import validate_scene_profile, scene_config

@pytest.fixture
def profile():
    return dict(name="Test camera", image_width=100, image_height=100,
                image_points=[[0,0],[99,0],[99,99],[0,99]],
                floor_points=[[0,0],[10,0],[10,6],[0,6]],bounds=[0,10,0,6],cell_size=.5,
                zones=[dict(zone_id="display",name="Display",polygon=[[0,0],[10,0],[10,6],[0,6]])],
                candidate_zone_ids=["display"])

def test_profile_is_isolated_and_camera_specific(profile):
    base={"zones":{"merl":[{"untouched":True}]}}
    before=copy.deepcopy(base)
    result=scene_config(base,profile)
    assert base==before
    assert result["zones"]["uploaded_scene"][0]["zone_id"]=="display"
    with pytest.raises(ValueError,match="does not match"):
        validate_scene_profile(profile,920,680)

@pytest.mark.parametrize("field,value",[("cell_size",0),("cell_size",float("nan")),
 ("bounds",[0,0,0,6]),("candidate_zone_ids",["unknown"]),("hysteresis_window",0),
 ("image_points",[[0,0],[1,1],[2,2],[3,3]])])
def test_profile_rejects_invalid_geometry(profile,field,value):
    profile[field]=value
    with pytest.raises(ValueError):validate_scene_profile(profile)

def test_overlapping_zones_rejected(profile):
    profile["zones"].append(dict(zone_id="other",name="Other",polygon=[[0,0],[5,0],[5,5],[0,5]]))
    with pytest.raises(ValueError,match="overlap"):validate_scene_profile(profile)

def test_explicit_candidate_zone_names(profile):
    from scripts.run_vscode_demo import select_exploratory_product_zone_candidate
    r=select_exploratory_product_zone_candidate(["display"],{"display":5},
        {"display":{"observed_dwell_span_sec":2,"episodes":1}},candidate_zone_ids=["display"])
    assert r["candidate_zone_id"]=="display"
    assert r["actionable_recommendation"]=="WITHHELD"

def test_empty_candidates_preserves_no_recommendation(profile):
    profile["candidate_zone_ids"]=[]
    assert validate_scene_profile(profile)["candidate_zone_ids"]==[]
