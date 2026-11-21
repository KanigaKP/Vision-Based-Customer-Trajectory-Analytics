"""
Direction A Destination 2: New Analysis View.
==============================================
Provides a structured 5-step analysis setup for local indoor video clips:
  Step 1: Upload Video & Representative Frame Preview (local-only, OpenCV decoding, frame slider)
  Step 2: Camera Profile Selection & Visual Single-Camera Calibration Editor (Image Quad P1..P4, Floor Plane Zones, Homography SVD diagnostics, JSON import/export)
  Step 3: Analysis Settings (Candidate objective, frame cap, invalidating stale state)
  Step 4: Execution (local YOLO11-n ONNX & ByteTrack pipeline with real-time progress)
  Step 5: Completed Results Handoff (direct link to inspect run in Results Workspace)

Clear distinction is maintained between active offline features and proposed capabilities.
All floor coordinates are reported in configured units, not physical surveyed metres.
"""

import os
import json
import uuid
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List

import streamlit as st

from dashboard.report_loader import resolve_demo_artifact_path
from dashboard.theme import (
    render_metric_card_html,
    render_withheld_banner_html,
    COLOR_TEAL_ACCENT,
)
from dashboard.visual_setup import (
    extract_representative_frame,
    create_default_profile,
    validate_and_diagnose_profile,
    export_profile_json,
    import_profile_json,
    render_visual_editor,
)


class LocalVideoFile:
    """Lightweight wrapper for existing local video files matching Streamlit upload interface."""

    def __init__(self, file_path: str):
        self.path = file_path
        self.name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            self._bytes = f.read()

    def getvalue(self) -> bytes:
        return self._bytes

    def getbuffer(self) -> memoryview:
        return memoryview(self._bytes)


def render_new_analysis_view(reports_dir: str = "outputs/reports") -> None:
    """Render the 5-step New Analysis workflow."""
    st.markdown(
        '<div class="dir-a-hero">'
        '<div class="dir-a-hero-eyebrow">Local Analysis Setup · Destination 2</div>'
        '<h1>New Video Analysis & Calibration Setup</h1>'
        '<p>'
        'Configure and execute an offline detection and tracking run for recorded indoor video. '
        'Visual single-camera setup maps image coordinates to configured floor units and defines '
        'analytic zones without manual JSON editing.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # Architectural & Methodological Boundaries
    st.info(
        "🔒 **Local-Only Offline Execution:** Video data is processed entirely on this device "
        "using YOLO11-n ONNX inference on CPU and ByteTrack. No live streams, remote APIs, "
        "or identity re-identification models are supported. Floor coordinates represent "
        "configured units, not independently surveyed physical metres."
    )

    # Capabilities Callout
    with st.expander("ℹ️ Calibration Interface Capabilities (Active vs. Proposed)"):
        st.markdown(
            "- **Active in this release:** Representative frame decoding via OpenCV, interactive visual 4-point perspective quad placement (P1..P4) on image canvas, "
            "floor-plane zone polygon drawing and vertex repositioning, numeric coordinate fine-tuning, real-time SVD homography validation, "
            "JSON import/export, and seamless handoff to offline detection and tracking.\n"
            "- **Proposed capabilities:** Live camera-overlay polygon drawing directly over streaming video and automatic corner feature detection.\n"
            "- **Mathematical disclaimer:** SVD planar homography computes mathematical projection from 4 correspondence points "
            "(validating SVD condition number, non-singularity $\\det(H) \\neq 0$, quad convexity, and non-collinearity); "
            "it does not constitute independently surveyed physical calibration or lens distortion correction. "
            "Floor plane projection measures ground foot-point contacts; it does not measure elevated countertop surfaces or installation clearance.\n"
            "- **Out of scope for this offline release:** Multi-camera re-identification, 3D LiDAR/depth sensor fusion, and live camera streams."
        )

    # -------------------------------------------------------------
    # Step 1: Upload Video & Local Preview
    # -------------------------------------------------------------
    st.markdown("### Step 1: Upload Video & Local Preview")
    video_source_mode = st.radio(
        "Video Source Selection",
        ["Select Verified Local Recording", "Upload Video File"],
        index=0,
        horizontal=True,
        key="new_analysis_video_source_mode",
        help="Select a local recording already bundled in the project or upload your own video.",
    )

    uploaded = None
    if video_source_mode == "Select Verified Local Recording":
        verified_candidates = [
            ("data/merl/10_1_crop.mp4", "MERL Shopping Aisle (data/merl/10_1_crop.mp4)"),
            ("data/prcv_site03_boutique.mp4", "PRCV Site 03 Home Decor Boutique (data/prcv_site03_boutique.mp4)"),
            ("data/prcv_site02_jewelry.mp4", "PRCV Site 02 Luxury Jewelry (data/prcv_site02_jewelry.mp4)"),
        ]
        available_clips = {label: path for path, label in verified_candidates if os.path.isfile(path)}
        if available_clips:
            selected_label = st.selectbox(
                "Choose Verified Local Recording",
                list(available_clips.keys()),
                key="new_analysis_verified_clip_select",
            )
            selected_path = available_clips[selected_label]
            uploaded = LocalVideoFile(selected_path)
            st.session_state["persisted_uploaded_file"] = uploaded
        else:
            st.warning("No verified local clips found in `data/`. Please upload a video file below.")
            raw_uploaded = st.file_uploader(
                "Select recorded indoor video clip (MP4, AVI, MOV, MKV)",
                type=["mp4", "avi", "mov", "mkv"],
                key="new_analysis_video_file",
            )
            if raw_uploaded is not None:
                st.session_state["persisted_uploaded_file"] = raw_uploaded
            uploaded = st.session_state.get("persisted_uploaded_file")
    else:
        raw_uploaded = st.file_uploader(
            "Select recorded indoor video clip (MP4, AVI, MOV, MKV)",
            type=["mp4", "avi", "mov", "mkv"],
            key="new_analysis_video_file",
        )
        if raw_uploaded is not None:
            st.session_state["persisted_uploaded_file"] = raw_uploaded
        uploaded = st.session_state.get("persisted_uploaded_file")

    indoor_confirmed = st.checkbox(
        "I confirm this is recorded indoor footage (caller metadata assertion)",
        value=True,
        key="new_analysis_indoor_confirm",
    )

    frame_info: Optional[Dict[str, Any]] = None

    if uploaded is not None:
        video_bytes = uploaded.getvalue()
        # Full content hash with byte length prevents different recordings with matching initial bytes from reusing state
        video_hash = hashlib.sha256(video_bytes).hexdigest()[:16] + f"_{len(video_bytes)}"
        
        # Invalidate previous state if video changed
        if st.session_state.get("new_analysis_video_hash") != video_hash:
            st.session_state["new_analysis_video_hash"] = video_hash
            st.session_state.pop("new_analysis_frame_info", None)
            st.session_state.pop("calib_profile", None)
            for editor_key in list(st.session_state):
                if editor_key.startswith("num_") or editor_key in ("calib_selected_zone", "calib_last_component_value", "calib_import_digest"):
                    st.session_state.pop(editor_key, None)
            st.session_state.pop("new_analysis_summary", None)
            st.session_state.pop("new_analysis_run_dir", None)
            st.session_state.pop("new_analysis_run_id", None)
            st.session_state["new_analysis_sample_idx"] = 0

        # Extract representative frame
        sample_idx = st.session_state.get("new_analysis_sample_idx", 0)
        cached_info = st.session_state.get("new_analysis_frame_info")

        if cached_info is None or cached_info.get("frame_idx") != sample_idx:
            try:
                frame_info = extract_representative_frame(video_bytes, frame_idx=sample_idx)
                st.session_state["new_analysis_frame_info"] = frame_info
            except Exception as exc:
                st.warning(f"Could not extract video frame preview: {exc}")
                frame_info = None
        else:
            frame_info = cached_info

        if frame_info:
            w = frame_info["width"]
            h = frame_info["height"]
            tot = frame_info["total_frames"]
            fps = frame_info["fps"]
            ts = frame_info["timestamp_sec"]

            st.caption(
                f"**Video Metadata:** `{uploaded.name}` · Dimensions: **{w} × {h} px** · "
                f"FPS: **{fps}** · Total frames: **{tot}** (~{tot/fps:.1f}s) · Sampled at frame **{frame_info['frame_idx']}** ({ts:.2f}s)"
            )

            if tot > 1:
                new_idx = st.slider(
                    "Representative Frame Sampling Index",
                    min_value=0,
                    max_value=max(1, tot - 1),
                    value=frame_info["frame_idx"],
                    step=max(1, tot // 50),
                    key="new_analysis_frame_slider",
                    help="Adjust slider to choose a clear, unoccluded frame for visual calibration.",
                )
                if new_idx != frame_info["frame_idx"]:
                    st.session_state["new_analysis_sample_idx"] = new_idx
                    st.rerun()

    st.markdown("---")

    # -------------------------------------------------------------
    # Step 2: Camera Profile Selection & Visual Calibration
    # -------------------------------------------------------------
    st.markdown("### Step 2: Camera Profile & Spatial Calibration")
    st.caption("Planar projection maps 2D image coordinates to configured floor units and defines analytic zones.")

    calib_mode = st.radio(
        "Select Calibration Source",
        [
            "Visual Single-Camera Setup (Interactive Editor)",
            "Use MERL Supermarket Preset (only for MERL 10_1_crop.mp4 camera view)",
            "Use PRCV Site03 Home Decor Boutique Preset (for prcv_site03_boutique.mp4 view)",
            "Use PRCV Site02 Luxury Jewelry Store Preset (for prcv_site02_jewelry.mp4 view)",
            "Upload Custom Camera Scene Profile (JSON)",
        ],
        key="new_analysis_calib_mode",
    )

    custom_profile: Optional[Dict[str, Any]] = None
    use_merl = (calib_mode == "Use MERL Supermarket Preset (only for MERL 10_1_crop.mp4 camera view)")
    profile_valid = True

    if use_merl:
        merl_profile_file = os.path.join("configs", "merl_scene_profile.json")
        if os.path.isfile(merl_profile_file):
            try:
                with open(merl_profile_file, "r", encoding="utf-8") as f_mp:
                    custom_profile = json.load(f_mp)
            except Exception:
                custom_profile = None
        st.success(
            "MERL scene profile selected: bounds X [0.0..10.0] × Y [0.0..6.0] in configured floor units. "
            "4 zones configured: `shelf_a`, `shelf_b`, `main_aisle`, `checkout`."
        )
        st.caption("Note: Floor scale in MERL preset is configured units, not independently surveyed metres.")

    elif calib_mode == "Use PRCV Site03 Home Decor Boutique Preset (for prcv_site03_boutique.mp4 view)":
        use_merl = False
        p_file = os.path.join("configs", "prcv_site03_scene_profile.json")
        if os.path.isfile(p_file):
            try:
                with open(p_file, "r", encoding="utf-8") as f_p:
                    custom_profile = json.load(f_p)
            except Exception:
                custom_profile = None
        st.success(
            "PRCV Site 03 Home Decor Boutique profile selected: bounds X [0.0..10.0] × Y [0.0..9.0] in configured floor units. "
            "4 zones configured: `side_display_left`, `center_aisle`, `display_table_center`, `checkout_counter`."
        )
        st.caption("Note: Ceiling-mounted overhead single-camera boutique view; floor coordinates are configured units.")

    elif calib_mode == "Use PRCV Site02 Luxury Jewelry Store Preset (for prcv_site02_jewelry.mp4 view)":
        use_merl = False
        p_file = os.path.join("configs", "prcv_site02_scene_profile.json")
        if os.path.isfile(p_file):
            try:
                with open(p_file, "r", encoding="utf-8") as f_p:
                    custom_profile = json.load(f_p)
            except Exception:
                custom_profile = None
        st.success(
            "PRCV Site 02 Luxury Jewelry Store profile selected: bounds X [0.0..8.0] × Y [0.0..9.0] in configured floor units. "
            "4 zones configured: `showcase_display_front`, `showcase_display_back`, `browsing_aisle`, `entrance_foyer`."
        )
        st.caption("Note: Ceiling-mounted overhead boutique view; floor coordinates are configured units.")

    elif calib_mode == "Upload Custom Camera Scene Profile (JSON)":
        st.write("Upload a validated JSON scene profile for this camera angle.")
        profile_file = st.file_uploader("Camera scene profile JSON", type=["json"], key="new_analysis_profile_file")
        if profile_file is not None:
            expected_w = frame_info["width"] if frame_info else None
            expected_h = frame_info["height"] if frame_info else None
            custom_profile, err = import_profile_json(profile_file.getvalue().decode("utf-8"), expected_w, expected_h)
            if err:
                profile_valid = False
                st.error(f"Scene profile validation error: {err}")
            else:
                st.success(f"Scene profile valid: `{custom_profile['name']}` ({len(custom_profile['zones'])} zones configured).")

    else:
        # VISUAL SINGLE-CAMERA SETUP (INTERACTIVE EDITOR)
        vid_w = frame_info["width"] if frame_info else 920
        vid_h = frame_info["height"] if frame_info else 680

        # Initialize profile state if missing or dimension mismatch
        curr_p = st.session_state.get("calib_profile")
        if curr_p is None or curr_p.get("image_width") != vid_w or curr_p.get("image_height") != vid_h:
            curr_p = create_default_profile(vid_w, vid_h)
            st.session_state["calib_profile"] = curr_p

        st.caption("Reference points must describe a visible floor plane in this recording. Starter zones are editable examples. Seated or heavily occluded scenes may support detection while providing insufficient floor-contact observations.")
        if st.button("Reset editor for this recording", key="reset_camera_editor"):
            st.session_state["calib_profile"] = create_default_profile(vid_w, vid_h)
            for editor_key in list(st.session_state):
                if editor_key.startswith("num_") or editor_key == "calib_last_component_value":
                    st.session_state.pop(editor_key, None)
            st.session_state["calib_editor_revision"] = st.session_state.get("calib_editor_revision", 0) + 1
            st.rerun()

        # Quick import / template controls
        c_io1, c_io2 = st.columns([3, 1])
        with c_io1:
            st.caption(
                f"Interactive editor configured for **{vid_w} × {vid_h} px**. "
                "Drag correspondence handles P1–P4 on the camera plane, or switch to the floor plane to position zones."
            )
        with c_io2:
            import_uploader = st.file_uploader("Import JSON", type=["json"], key="calib_import_uploader", label_visibility="collapsed")
            if import_uploader is not None and hashlib.sha256(import_uploader.getvalue()).hexdigest() != st.session_state.get("calib_import_digest"):
                imported, err = import_profile_json(import_uploader.getvalue().decode("utf-8"), vid_w, vid_h)
                if err:
                    st.error(f"Import failed: {err}")
                else:
                    st.session_state["calib_profile"] = imported
                    st.session_state["calib_import_digest"] = hashlib.sha256(import_uploader.getvalue()).hexdigest()
                    for editor_key in list(st.session_state):
                        if editor_key.startswith("num_") or editor_key == "calib_last_component_value":
                            st.session_state.pop(editor_key, None)
                    st.session_state["calib_editor_revision"] = st.session_state.get("calib_editor_revision", 0) + 1
                    curr_p = imported
                    st.success("Imported profile successfully!")
                    st.rerun()

        # Canvas mode switcher
        editor_mode_choice = st.radio(
            "Editor Active View",
            ["📷 Camera Plane (Image Points P1–P4)", "📐 Floor Plane (Analytical Zones)"],
            horizontal=True,
            key="calib_editor_tab",
        )
        editor_mode = "camera" if "Camera" in editor_mode_choice else "floor"

        # Render Interactive Visual Component
        comp_result = render_visual_editor(
            profile=curr_p,
            frame_info=frame_info,
            mode=editor_mode,
            selected_zone_id=st.session_state.get("calib_selected_zone", "shelf_a"),
            key=f"calib_visual_editor_{st.session_state.get('new_analysis_video_hash', 'empty')}_{st.session_state.get('calib_editor_revision', 0)}",
        )

        if comp_result and isinstance(comp_result, dict) and comp_result != st.session_state.get("calib_last_component_value"):
            st.session_state["calib_last_component_value"] = comp_result
            if comp_result.get("action") == "update_profile" and "profile" in comp_result:
                st.session_state["calib_profile"] = comp_result["profile"]
                curr_p = comp_result["profile"]
                # Synchronize widget state caches so number_input doesn't clobber dragged coordinates
                for i, pt in enumerate(curr_p.get("image_points", [])):
                    st.session_state[f"num_ix_{i}"] = float(pt[0])
                    st.session_state[f"num_iy_{i}"] = float(pt[1])
                for i, pt in enumerate(curr_p.get("floor_points", [])):
                    st.session_state[f"num_fx_{i}"] = float(pt[0])
                    st.session_state[f"num_fy_{i}"] = float(pt[1])
            elif comp_result.get("action") == "select_zone" and "selected_zone_id" in comp_result:
                st.session_state["calib_selected_zone"] = comp_result["selected_zone_id"]

        # Numeric Coordinate Editing Alternative (Full Accessibility)
        with st.expander("🔢 Numeric Coordinate Editor & Zone Properties", expanded=False):
            st.markdown("#### 1. Planar Homography Correspondence Points (P1–P4)")
            st.caption("P1: Bottom-Left, P2: Bottom-Right, P3: Top-Right, P4: Top-Left")
            
            p_cols = st.columns(4)
            labels = ["P1 (Bottom-Left)", "P2 (Bottom-Right)", "P3 (Top-Right)", "P4 (Top-Left)"]
            updated_img_pts = list(curr_p.get("image_points", []))
            updated_flr_pts = list(curr_p.get("floor_points", []))

            for i in range(4):
                with p_cols[i]:
                    st.markdown(f"**{labels[i]}**")
                    ix = st.number_input(f"Img X (px)", 0.0, float(vid_w), float(updated_img_pts[i][0]), step=5.0, key=f"num_ix_{i}")
                    iy = st.number_input(f"Img Y (px)", 0.0, float(vid_h), float(updated_img_pts[i][1]), step=5.0, key=f"num_iy_{i}")
                    fx = st.number_input(f"Floor X (units)", -50.0, 50.0, float(updated_flr_pts[i][0]), step=0.5, key=f"num_fx_{i}")
                    fy = st.number_input(f"Floor Y (units)", -50.0, 50.0, float(updated_flr_pts[i][1]), step=0.5, key=f"num_fy_{i}")
                    updated_img_pts[i] = [ix, iy]
                    updated_flr_pts[i] = [fx, fy]

            curr_p["image_points"] = updated_img_pts
            curr_p["floor_points"] = updated_flr_pts

            st.markdown("---")
            st.markdown("#### 2. Zone Geometry & Candidate Assignment")
            zones = curr_p.get("zones", [])
            zone_names = [z.get("name", z.get("zone_id")) for z in zones]
            selected_zone_idx = st.selectbox(
                "Select Zone to Configure",
                range(len(zone_names)),
                format_func=lambda i: f"{zone_names[i]} ({zones[i]['zone_id']})",
                key="num_zone_sel",
            ) if zones else None

            if selected_zone_idx is not None and selected_zone_idx < len(zones):
                target_zone = zones[selected_zone_idx]
                if st.session_state.get("numeric_zone_id") != target_zone["zone_id"]:
                    for widget_key in list(st.session_state):
                        if widget_key.startswith(("num_z_", "num_vx_", "num_vy_")):
                            st.session_state.pop(widget_key, None)
                    st.session_state["numeric_zone_id"] = target_zone["zone_id"]
                zc1, zc2, zc3 = st.columns([2, 1, 1])
                with zc1:
                    target_zone["name"] = st.text_input("Zone Display Name", target_zone.get("name", ""), key="num_z_name")
                with zc2:
                    target_zone["color"] = st.color_picker("Zone Color", target_zone.get("color", "#3498DB"), key="num_z_color")
                with zc3:
                    cand_ids = set(curr_p.get("candidate_zone_ids", []))
                    is_cand = target_zone["zone_id"] in cand_ids
                    new_is_cand = st.checkbox("Candidate Zone", is_cand, key="num_z_cand", help="Mark as candidate zone for layout optimization.")
                    if new_is_cand and target_zone["zone_id"] not in cand_ids:
                        cand_ids.add(target_zone["zone_id"])
                    elif not new_is_cand and target_zone["zone_id"] in cand_ids:
                        cand_ids.remove(target_zone["zone_id"])
                    curr_p["candidate_zone_ids"] = list(cand_ids)

                st.markdown("**Polygon Vertices (Floor Coordinates [X, Y]):**")
                poly = target_zone.get("polygon", [])
                v_cols = st.columns(min(len(poly), 4))
                for v_i in range(len(poly)):
                    col_idx = v_i % 4
                    with v_cols[col_idx]:
                        vx = st.number_input(f"V{v_i+1} X", -50.0, 50.0, float(poly[v_i][0]), step=0.2, key=f"num_vx_{v_i}")
                        vy = st.number_input(f"V{v_i+1} Y", -50.0, 50.0, float(poly[v_i][1]), step=0.2, key=f"num_vy_{v_i}")
                        poly[v_i] = [vx, vy]
                target_zone["polygon"] = poly

            # Add / Remove zone actions
            za1, za2 = st.columns(2)
            with za1:
                if st.button("➕ Add New Zone", key="btn_add_zone"):
                    new_id = f"zone_{len(zones)+1}"
                    new_z = {
                        "zone_id": new_id,
                        "name": f"New Zone {len(zones)+1}",
                        "polygon": [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]],
                        "color": "#9B59B6",
                    }
                    curr_p["zones"].append(new_z)
                    st.rerun()
            with za2:
                if len(zones) > 1 and selected_zone_idx is not None:
                    if st.button("🗑️ Delete Selected Zone", key="btn_del_zone"):
                        del_id = zones[selected_zone_idx]["zone_id"]
                        curr_p["zones"].pop(selected_zone_idx)
                        curr_p["candidate_zone_ids"] = [cid for cid in curr_p.get("candidate_zone_ids", []) if cid != del_id]
                        st.rerun()

        # Real-Time Validation & Homography Diagnostics
        is_valid, val_err, val_diag = validate_and_diagnose_profile(curr_p, video_width=vid_w, video_height=vid_h)

        if is_valid and val_diag:
            profile_valid = True
            custom_profile = curr_p
            cond_num = val_diag.get("condition_number", 0.0)
            repr_err = val_diag.get("mean_reprojection_error", 0.0)

            st.success(
                f"✅ **Geometry checks passed:** Planar Homography SVD Condition Number = **{cond_num:.2f}** · "
                f"Mean Reprojection Error = **{repr_err:.4f} px** · Zones: **{len(curr_p['zones'])}** · "
                f"Candidate Zones: **{len(curr_p['candidate_zone_ids'])}**"
            )
            st.caption(
                f"Floor Bounds: X [{curr_p['bounds'][0]}..{curr_p['bounds'][1]}] × "
                f"Y [{curr_p['bounds'][2]}..{curr_p['bounds'][3]}] configured units · Cell Size: {curr_p.get('cell_size', 0.5)} units. "
                "*(Note: Mathematical projection does not constitute surveyed physical calibration).* "
            )

            # Export JSON button
            st.download_button(
                "💾 Download Validated Scene Profile (JSON)",
                export_profile_json(curr_p),
                file_name=f"{curr_p.get('name', 'custom_scene').lower().replace(' ', '_')}.json",
                mime="application/json",
                key="calib_download_json",
            )
        else:
            profile_valid = False
            custom_profile = None
            st.error(f"❌ **Invalid Camera Profile:** {val_err}")
            st.caption("Actionable guidance: Ensure image points form a non-degenerate quad inside video dimensions, and all zone polygons are strictly convex without self-intersections.")

    st.markdown("---")

    # -------------------------------------------------------------
    # Step 3: Analysis Settings & Stale Invalidation
    # -------------------------------------------------------------
    st.markdown("### Step 3: Analysis Settings")
    c_set1, c_set2 = st.columns(2)
    with c_set1:
        candidate_objective_label = st.selectbox(
            "Exploratory Candidate Objective",
            ["Observed dwell span", "Sampled in-zone points", "Observed visit episodes"],
            help="Selects metric used to rank configured shelf zones. Exploratory traffic indicator only.",
            key="new_analysis_candidate_obj",
        )
        candidate_objective = {
            "Observed dwell span": "observed_dwell",
            "Sampled in-zone points": "sampled_points",
            "Observed visit episodes": "visit_episodes",
        }[candidate_objective_label]

    with c_set2:
        frame_sampling_cap = st.number_input(
            "Sampling Frame Cap (Max sampled frames)",
            min_value=30,
            max_value=2000,
            value=700,
            step=50,
            key="new_analysis_frame_cap",
            help="Full MERL recording requires ~633 sampled frames at 5 FPS.",
        )

    # Invalidate stale results when parameters change
    if uploaded is not None:
        analysis_key = hashlib.sha256(uploaded.getbuffer()).hexdigest() + repr(
            (use_merl, candidate_objective, frame_sampling_cap, json.dumps(custom_profile, sort_keys=True) if custom_profile else "")
        )
        if st.session_state.get("new_analysis_key") != analysis_key:
            st.session_state.pop("new_analysis_summary", None)
            st.session_state.pop("new_analysis_run_dir", None)
            st.session_state.pop("new_analysis_run_id", None)
    else:
        analysis_key = ""

    st.markdown("---")

    # -------------------------------------------------------------
    # Step 4: Execution
    # -------------------------------------------------------------
    st.markdown("### Step 4: Execution")
    if uploaded is None:
        st.info("Please upload an indoor video clip in Step 1 to proceed.")
        return

    if not indoor_confirmed:
        st.warning("Please confirm the indoor footage checkbox in Step 1.")
        return

    if not profile_valid or (not use_merl and custom_profile is None):
        st.warning("Please configure or import a valid camera calibration profile in Step 2.")
        return

    if st.button("🚀 Analyze Video (Run Pipeline)", type="primary", key="new_analysis_run_btn", use_container_width=True):
        st.session_state.pop("new_analysis_summary", None)
        st.session_state.pop("new_analysis_run_dir", None)
        st.session_state.pop("new_analysis_run_id", None)
        
        try:
            import yaml
            from scripts.run_vscode_demo import run_real_data_demo

            run_id = f"upload_{uuid.uuid4().hex[:8]}"
            run_dir = os.path.join(reports_dir, "uploaded_runs", run_id)
            os.makedirs(run_dir, exist_ok=True)
            
            suffix = Path(uploaded.name).suffix.lower()
            if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
                st.error("Unsupported video extension.")
                return
                
            input_path = os.path.join(run_dir, f"input_video{suffix}")
            with open(input_path, "wb") as f_in:
                f_in.write(uploaded.getvalue())

            with open("config.yaml", "r", encoding="utf-8") as f_cfg:
                config = yaml.safe_load(f_cfg)

            progress_bar = st.progress(0.0, text="Initializing YOLO11-n ONNX and ByteTrack...")

            def update_progress(val: float, count: int) -> None:
                progress_bar.progress(val, text=f"Processing sampled frame {count}...")

            with st.spinner("Executing detection, tracking, homography, and zone analytics..."):
                summary = run_real_data_demo(
                    config,
                    reports_dir=run_dir,
                    max_frames=int(frame_sampling_cap),
                    video_path=input_path,
                    progress_callback=update_progress,
                    artifact_prefix="run",
                    use_merl_profile=use_merl or (custom_profile is not None),
                    candidate_objective=candidate_objective,
                    scene_profile=custom_profile,
                )

            st.session_state["new_analysis_key"] = analysis_key
            st.session_state["new_analysis_summary"] = summary
            st.session_state["new_analysis_run_dir"] = run_dir
            st.session_state["new_analysis_run_id"] = run_id
            st.session_state["authoritative_run_id"] = f"uploaded_runs/{run_id}/run_summary.json"
            st.success("Analysis pipeline completed successfully!")

        except Exception as exc:
            st.error(f"Execution failed: {type(exc).__name__}: {exc}")
            return

    # -------------------------------------------------------------
    # Step 5: Completed Results Handoff
    # -------------------------------------------------------------
    summary = st.session_state.get("new_analysis_summary")
    run_dir = st.session_state.get("new_analysis_run_dir")
    run_id = st.session_state.get("new_analysis_run_id")

    if summary and run_dir and run_id:
        st.markdown("---")
        st.markdown("### Step 5: Completed Analysis Results & Handoff")
        st.success(f"Run ID `{run_id}` has been saved and is ready for exploration.")

        # Metric summary
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(render_metric_card_html("Sampled Frames", str(summary.get("total_frames_evaluated", 0))), unsafe_allow_html=True)
        with m2:
            st.markdown(render_metric_card_html("Detections", str(summary.get("total_detections", 0))), unsafe_allow_html=True)
        with m3:
            st.markdown(render_metric_card_html("Clip Tracks", str(summary.get("total_tracks", 0))), unsafe_allow_html=True)
        with m4:
            cand = summary.get("exploratory_layout_candidate") or {}
            cand_zone = cand.get("candidate_zone_id") or "No candidate selected"
            st.markdown(render_metric_card_html("Candidate Zone", cand_zone, "Exploratory only"), unsafe_allow_html=True)

        st.markdown(
            '<div style="background:#F0FDFA; border:1px solid #99F6E4; border-radius:10px; padding:1.2rem; margin:1rem 0;">'
            '<h4 style="color:#0D9488; margin:0 0 0.5rem 0;">Ready to inspect detailed spatial analytics?</h4>'
            '<p style="color:#134E4A; margin:0 0 1rem 0;">'
            'Open this run in the <strong>Results Workspace</strong> to review tracking video playback, '
            'orthogonal floor plane trajectories, discrete density heatmaps, dwell breakdowns, and transition matrices.'
            '</p></div>',
            unsafe_allow_html=True,
        )

        c_act1, c_act2 = st.columns(2)
        with c_act1:
            if st.button("🎨 Open Run in Spatial Analysis Studio ➔", type="primary", use_container_width=True, key="new_analysis_open_studio_btn"):
                st.session_state["authoritative_run_id"] = f"uploaded_runs/{run_id}/run_summary.json"
                st.session_state["studio_entry_mode"] = "🔍 Explore Existing Recording & Evaluation Stages"
                st.session_state["studio_active_stage"] = 6
                st.session_state["pending_navigation_view"] = "Spatial Analysis Studio"
                st.rerun()
        with c_act2:
            if st.button("📊 Open Run in Results Workspace ➔", use_container_width=True, key="new_analysis_open_workspace_btn"):
                st.session_state["authoritative_run_id"] = f"uploaded_runs/{run_id}/run_summary.json"
                st.session_state["pending_navigation_view"] = "Results Workspace"
                st.rerun()
