/**
 * Spatial Player Engine — Production 3D Analytical & Projection Engine.
 * ====================================================================
 * Shared mathematical and state routines for the synchronized retail player.
 * Used directly in dashboard/components/synchronized_player/index.html
 * and executed directly in automated regression test suites.
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    // Node.js / CommonJS
    module.exports = factory();
  } else {
    // Browser global
    root.SpatialPlayerEngine = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /**
   * Dynamic Scale Handling: Compute adequate labelled seconds scale ceiling for dwell extrusion.
   * Dwell values 45s, 85s, 135s produce ceilings 50s, 100s, 150s.
   *
   * @param {Object} zoneDwell - Mapping of zone_id -> dwell info dict.
   * @returns {number} maxScaleDwellSec ceiling.
   */
  function computeMaxScaleDwell(zoneDwell) {
    let maxObservedDwell = 0.0;
    if (zoneDwell && typeof zoneDwell === 'object') {
      Object.keys(zoneDwell).forEach(function (zid) {
        const zd = zoneDwell[zid];
        if (!zd) return;
        const d = zd.observed_dwell_sec !== undefined ? zd.observed_dwell_sec : zd.observed_dwell_span_sec;
        if (typeof d === 'number' && isFinite(d) && d > maxObservedDwell) {
          maxObservedDwell = d;
        }
      });
    }

    if (maxObservedDwell <= 20.0) {
      return 20.0;
    }
    const step = maxObservedDwell > 100 ? 50 : (maxObservedDwell > 50 ? 25 : 10);
    return Math.ceil(maxObservedDwell / step) * step;
  }

  /**
   * Compute graduated ruler ticks for vertical dwell seconds ruler.
   *
   * @param {number} maxScaleDwellSec - The active scale ceiling.
   * @param {number} numIntervals - Desired number of intervals (default 4).
   * @returns {Array<{val: number, label: string, normZ: number}>} Tick descriptors.
   */
  function computeRulerTicks(maxScaleDwellSec, numIntervals) {
    const intervals = numIntervals || 4;
    const ticks = [];
    for (let i = 0; i <= intervals; i++) {
      const val = (maxScaleDwellSec / intervals) * i;
      ticks.push({
        val: val,
        label: (val % 1 === 0 ? val.toFixed(0) : val.toFixed(1)) + 's',
        normZ: val / maxScaleDwellSec
      });
    }
    return ticks;
  }

  /**
   * Classify dwell state to distinguish positive dwell, evaluated zero, and missing dwell.
   *
   * @param {number|null|undefined} dwellSec - Dwell duration in seconds.
   * @param {string|null|undefined} dwellStatus - Status string from report ('evaluated_zero', 'missing', etc.).
   * @returns {{isRecorded: boolean, isZero: boolean, isMissing: boolean, state: string, label: string}}
   */
  function classifyDwellState(dwellSec, dwellStatus) {
    const hasNum = (typeof dwellSec === 'number' && isFinite(dwellSec));
    const isRecorded = hasNum && dwellSec > 0;
    const isZero = (hasNum && dwellSec === 0) || dwellStatus === 'evaluated_zero';
    const isMissing = !isRecorded && !isZero;

    let state = 'missing';
    let label = '(unrecorded)';
    if (isRecorded) {
      state = 'positive';
      label = dwellSec.toFixed(1) + 's (observed)';
    } else if (isZero) {
      state = 'evaluated_zero';
      label = '0.0s (eval zero)';
    }

    return {
      isRecorded: isRecorded,
      isZero: isZero,
      isMissing: isMissing,
      state: state,
      label: label
    };
  }

  /**
   * Segment trajectory history into continuous trails breaking across unobserved intervals.
   * Never draws continuous motion through unobserved sample gaps.
   *
   * @param {Array<{sampleIdx: number, pt: Array<number>}>} points - Chronological track points.
   * @returns {Array<Array<{sampleIdx: number, pt: Array<number>}>>} Contiguous segments of length >= 2.
   */
  function segmentTrajectoryTrails(points) {
    if (!Array.isArray(points) || points.length < 2) {
      return [];
    }

    const segments = [];
    let currentSegment = [points[0]];

    for (let j = 1; j < points.length; j++) {
      if (points[j].sampleIdx === points[j - 1].sampleIdx + 1) {
        currentSegment.push(points[j]);
      } else {
        // Gap encountered: break segment
        if (currentSegment.length >= 2) {
          segments.push(currentSegment);
        }
        currentSegment = [points[j]];
      }
    }

    if (currentSegment.length >= 2) {
      segments.push(currentSegment);
    }

    return segments;
  }

  /**
   * Project 3D floor point into 2D canvas coordinates using camera model.
   *
   * @param {number} x - Floor X.
   * @param {number} y - Floor Y.
   * @param {number} z - Floor Z (dwell height).
   * @param {number} cx - Floor center X.
   * @param {number} cy - Floor center Y.
   * @param {number} S - View scale.
   * @param {number} W - Canvas width.
   * @param {number} H - Canvas height.
   * @param {Object} camera - Camera state { yaw, pitch, panX, panY, scy }.
   * @returns {{x: number, y: number, depth: number}} Projected 2D canvas point.
   */
  function project3D(x, y, z, cx, cy, S, W, H, camera) {
    const dx = x - cx;
    const dy = y - cy;
    const yaw = camera.yaw;
    const pitch = camera.pitch;
    const panX = camera.panX || 0;
    const panY = camera.panY || 0;
    const scy = camera.scy || 0;

    // Azimuth rotation around vertical axis
    const rx = dx * Math.cos(yaw) - dy * Math.sin(yaw);
    const ry = dx * Math.sin(yaw) + dy * Math.cos(yaw);

    // Elevation pitch tilt
    const sx = rx;
    const sy = -(ry * Math.sin(pitch) + z * Math.cos(pitch));
    const depth = ry * Math.cos(pitch) - z * Math.sin(pitch);

    return {
      x: W / 2 + panX + sx * S,
      y: H / 2 + panY + (sy - scy) * S,
      depth: depth
    };
  }

  // Explicit commands are deduplicated by request identity, never by destination.
  function createSeekGate() {
    let lastToken, lastSample;
    return function(sample, token) {
      if (!Number.isInteger(sample) || sample < 0) return false;
      if (token !== undefined && token !== null) {
        if (token === lastToken) return false;
        lastToken = token;
      } else {
        if (sample === lastSample) return false;
      }
      lastSample = sample;
      return true;
    };
  }

  return {
    createSeekGate: createSeekGate,
    computeMaxScaleDwell: computeMaxScaleDwell,
    computeRulerTicks: computeRulerTicks,
    classifyDwellState: classifyDwellState,
    segmentTrajectoryTrails: segmentTrajectoryTrails,
    project3D: project3D
  };
}));
