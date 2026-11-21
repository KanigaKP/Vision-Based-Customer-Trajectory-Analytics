"""Validated camera-specific input boundary; never upgrades empirical evidence."""
import copy
import math
import re
import cv2
import numpy as np
from src.spatial import HomographyCalibrator


def validate_scene_profile(profile, width=None, height=None):
    if not isinstance(profile, dict):
        raise ValueError("Scene profile must be a JSON object")
    p = copy.deepcopy(profile)
    if not isinstance(p.get("name"), str) or not p["name"].strip():
        raise ValueError("Provide a scene name")
    for key, actual in [("image_width", width), ("image_height", height)]:
        if type(p.get(key)) is not int or p[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
        if actual is not None and p[key] != actual:
            raise ValueError(f"{key} does not match the uploaded video's {actual} pixels")
    def points(value, minimum):
        a = np.asarray(value, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < minimum or not np.isfinite(a).all():
            raise ValueError("Coordinates must be finite x,y pairs")
        return a
    bounds = np.asarray(p.get("bounds"), dtype=float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError("bounds must be [min_x,max_x,min_y,max_y]")
    x0,x1,y0,y1 = bounds
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Floor bounds must have positive width and height")
    cell = p.get("cell_size")
    if isinstance(cell, bool) or not isinstance(cell, (int,float)) or not math.isfinite(cell) or cell <= 0:
        raise ValueError("cell_size must be finite and positive")
    if math.ceil((x1-x0)/cell)*math.ceil((y1-y0)/cell) > 10000:
        raise ValueError("Use at most 10,000 grid cells for this local prototype")
    src, dst = points(p.get("image_points"),4), points(p.get("floor_points"),4)
    if len(src) != len(dst):
        raise ValueError("Image and floor correspondence counts must match")
    if np.any(src < 0) or np.any(src[:,0] >= p["image_width"]) or np.any(src[:,1] >= p["image_height"]):
        raise ValueError("Image correspondences must lie inside the video frame")
    def inside(a):
        return np.all((a[:,0]>=x0)&(a[:,0]<=x1)&(a[:,1]>=y0)&(a[:,1]<=y1))
    if not inside(dst):
        raise ValueError("Floor correspondences must lie within the declared bounds")
    HomographyCalibrator(src.tolist(),dst.tolist(),unit="configured_units",dataset_name=p["name"])
    zones=p.get("zones")
    if not isinstance(zones,list) or not zones or len(zones)>50:
        raise ValueError("Provide 1 to 50 zones")
    ids=set(); polygons=[]
    for zone in zones:
        zid=zone.get("zone_id")
        if not isinstance(zid,str) or not re.fullmatch(r"[a-zA-Z0-9_-]+",zid) or zid in ids:
            raise ValueError("Zone IDs must be unique letters, digits, underscores or hyphens")
        ids.add(zid)
        if not isinstance(zone.get("name"),str) or not zone["name"].strip():
            raise ValueError("Each zone requires a display name")
        poly=points(zone.get("polygon"),3).astype(np.float32)
        if not inside(poly) or not cv2.isContourConvex(poly) or abs(cv2.contourArea(poly))<1e-6:
            raise ValueError("Zones must be nondegenerate convex polygons inside the bounds")
        for previous in polygons:
            if cv2.intersectConvexConvex(previous,poly)[0] > 1e-6:
                raise ValueError("Zone interiors must not overlap; shared edges are allowed")
        polygons.append(poly)
        if not re.fullmatch(r"#[0-9a-fA-F]{6}",zone.get("color","#3498db")):
            raise ValueError("Zone color must be a six-digit hex color")
    candidates=p.get("candidate_zone_ids")
    if not isinstance(candidates,list) or any(not isinstance(z,str) or z not in ids for z in candidates) or len(set(candidates))!=len(candidates):
        raise ValueError("candidate_zone_ids must list distinct configured zone IDs")
    window=p.get("hysteresis_window",3)
    if type(window) is not int or not 1<=window<=120:
        raise ValueError("hysteresis_window must be an integer from 1 to 120")
    p["hysteresis_window"]=window
    return p


def scene_config(base, profile):
    p=validate_scene_profile(profile)
    config=copy.deepcopy(base)
    config.setdefault("homography",{})["uploaded_scene"]={
        "image_points":p["image_points"],"floor_points":p["floor_points"],
        "unit":"configured_units","coordinate_type":"caller_defined_floor_grid"}
    config.setdefault("zones",{})["uploaded_scene"]=p["zones"]
    config["zones"]["hysteresis_window"]=p["hysteresis_window"]
    return config
