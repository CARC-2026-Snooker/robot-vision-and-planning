"""
ball_align.py — Back-camera ball alignment for claw pickup.

Runs independently. Opens the back camera, finds the nearest white ball
on the blue podium, and broadcasts alignment state + movement commands
so mission_controller can drive the robot into position.

How to use from mission_controller:
    align = BallAlignState()          # start listening
    do_ball_align(mc, align)          # drive until aligned
    align.stop()

Or run standalone for tuning:
    uv run ball_align.py

Coordinate conventions (back camera, robot facing 270° = toward tags):
  The back camera faces the OPPOSITE direction from the tags (toward ~90°).
  LEFT/RIGHT in the camera image directly corresponds to world LEFT/RIGHT
  because the camera is rear-facing (not mirrored for this purpose).

  lateral_err_px > 0  = ball is RIGHT of frame centre → strafe RIGHT
  lateral_err_px < 0  = ball is LEFT  of frame centre → strafe LEFT
  distance_state = "FAR"  → robot needs to move BACK (toward ball/pedestal)
  distance_state = "CLOSE" → distance is good

Alignment is declared when:
  - |lateral_err_px| < LATERAL_TOL_PX_ALIGN  (horizontal centred)
  - distance_state == "CLOSE"                 (robot close enough)
"""

import cv2
import sys
import os
import time
import math
import socket
import json
import threading
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CAM_BACK,
    TELEMETRY_PORT,
    MOVEMENT_HOST, MOVEMENT_CMD_PORT,
    MOVE_POWER_MIN, MOVE_POWER_MAX,
    ROT_POWER_MIN,  ROT_POWER_MAX,
    MOVE_DEAD_MM, ROT_DEAD_DEG,
    DISPLAY_SCALE,
)

HAS_DISPLAY = bool(os.environ.get("DISPLAY"))

DISPLAY_SCALE = 1.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TUNABLE PARAMETERS
# All values calibrated for the camera view shown in the reference images.
# Tune on the actual robot by running standalone: uv run ball_align.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# AT UNI

# HSV_BALL_LO = np.array([0, 0, 43])
# HSV_BALL_HI = np.array([179, 83, 255])

# # pedestal  (1825 samples)
# HSV_PEDESTAL_LO = np.array([86, 38, 0])
# HSV_PEDESTAL_HI = np.array([153, 213, 255])

# White ball HSV thresholds.
# White = very low saturation, very high brightness.
WHITE_H_MIN, WHITE_H_MAX = 0,  180
WHITE_S_MIN, WHITE_S_MAX = 0,  80
# WHITE_V_MIN, WHITE_V_MAX = 43, 255
WHITE_V_MIN, WHITE_V_MAX = 150, 255

# Blue podium HSV thresholds.
# Used to reject white blobs that are NOT above blue (e.g. white floor).
BLUE_H_MIN, BLUE_H_MAX = 86,  153
BLUE_S_MIN, BLUE_S_MAX = 45,  255
BLUE_V_MIN, BLUE_V_MAX = 50,  255

# AT COMPT

# # White ball HSV thresholds.
# # White = very low saturation, very high brightness.
# WHITE_H_MIN, WHITE_H_MAX = 0,  180
# WHITE_S_MIN, WHITE_S_MAX = 0, 255
# WHITE_V_MIN, WHITE_V_MAX = 80,  255

# # Blue podium HSV thresholds.
# # Used to reject white blobs that are NOT above blue (e.g. white floor).
# BLUE_H_MIN, BLUE_H_MAX = 60,  160
# BLUE_S_MIN, BLUE_S_MAX = 5, 255
# BLUE_V_MIN, BLUE_V_MAX = 80,  204

# Fraction of frame height to search for balls.
# Ignore the bottom portion to avoid seeing the white floor.
# Balls sit on the blue pedestal which occupies the lower half,
# but the BALL itself is in the upper portion.
# Setting 0.80 means we look in the top 80% of frame for white blobs.
SEARCH_TOP_FRACTION = 1.0

# Minimum blob area in pixels (at the camera's capture resolution).
# Avoids picking up small reflections or noise.
MIN_BALL_AREA_PX = 3000

# Minimum blue pedestal blob area in pixels.
# Filters out tiny blue noise specks that would set a bogus cutoff line.
# The pedestal should be a large-ish blob; raise if false detections persist.
MIN_PEDESTAL_AREA_PX = 7000

# Circularity filter for white ball blobs.
# circularity = 4π·area / perimeter²  →  1.0 = perfect circle, 0 = very elongated.
#
# Two-stage threshold based on blob area (area ≈ proximity):
#   FAR  (small blob)  — ball is fully visible → enforce strict roundness.
#                        Floor / walls are flat rectangles (circ << 0.4) so a
#                        high threshold cleanly rejects them.
#   CLOSE (large blob) — robot is near the pedestal; the bottom of the ball is
#                        cut off by the pedestal edge, leaving a half-circle or
#                        D-shape (circ ≈ 0.39–0.55).  Relax the threshold so we
#                        keep tracking without losing the ball just as we arrive.
#
# CIRC_SWITCH_AREA_PX:  blob area above which we switch to the relaxed threshold.
#                       Tune: run ball_align.py standalone, note the area shown
#                       when the ball starts getting clipped, set this ~20% below.
MIN_BALL_CIRCULARITY_FAR   = 0.75   # strict  — full ball visible (far)
MIN_BALL_CIRCULARITY_CLOSE = 0.55   # relaxed — ball bottom cut off (close)
CIRC_SWITCH_AREA_PX        = 80_000 # px² — switch to relaxed above this area

# ── Temporal stability / flicker suppression ──────────────────────────────────
# A new detection state must appear this many consecutive frames before it is
# accepted.  Prevents single-frame noise from causing flickering state changes.
#
# DETECT_CONFIRM_FRAMES: frames a NEW ball/pedestal detection must persist
#   before switching from LOST → detected.
# DETECT_LOSS_FRAMES: frames detection must be ABSENT before switching to LOST
#   (hysteresis — once found, we're reluctant to drop it).
# STATE_CONFIRM_FRAMES: frames a NEW distance_state ("FAR"/"CLOSE") or x_cmd
#   must persist before it is passed out.  Smooths command jitter.
DETECT_CONFIRM_FRAMES = 3   # frames needed to confirm a new detection
DETECT_LOSS_FRAMES    = 5   # frames of absence before declaring LOST
STATE_CONFIRM_FRAMES  = 3   # frames needed for a state/command change to stick

# Target area in pixels for "CLOSE" distance state.
# When the ball's white area exceeds this, the robot is close enough.
# Tune by observing what area you see when the claw is at pickup distance.
# From image 1 (backed up): ball fills ~40% of frame → ~0.4 * 640*480 ≈ 120,000px
# From image 2 (aligned):   ball fills ~50% of frame edge → much larger
# Start conservative:
AREA_CLOSE_THRESHOLD = 280_000   # px² — tune upward if robot stops too far away

# Lateral tolerance: ball centroid within this many pixels of frame centre = aligned.
LATERAL_TOL_PX_ALIGN = 30    # px

# Pedestal-fallback lateral tolerance — slightly looser than for the ball itself.
# When no white ball is visible we just want the pedestal roughly centred so the
# ball will appear once the robot slides into range.
PEDESTAL_LATERAL_TOL_PX = 60   # px — tune looser than LATERAL_TOL_PX_ALIGN

# Movement speeds for alignment (sent to movement_server via UDP)
ALIGN_STRAFE_POWER = 50   # power for LEFT/RIGHT strafe during alignment
ALIGN_BACK_POWER   = 200   # power for BACK movement during alignment
ALIGN_PEDESTAL_STRAFE_POWER = 40  # power when strafing toward pedestal (slower, gentler)

# ── Perpendicularity correction ───────────────────────────────────────────────
# The robot must approach the pedestal straight-on (perpendicular) so the claw
# grabs the ball cleanly.  We measure the "skew" of the blue pedestal's top
# edge: if the robot is angled, one side of the pedestal appears higher in the
# image than the other.
#
#   skew_px = right_top_y - left_top_y  (y=0 is TOP in OpenCV)
#     > 0  →  right side appears lower (further away)  →  robot angled LEFT   →  TURN RIGHT
#     < 0  →  left  side appears lower                 →  robot angled RIGHT  →  TURN LEFT
#     ≈ 0  →  top edge is horizontal  →  robot is perpendicular  ✓
#
# PERP_TOL_PX:  skew below this is treated as "perpendicular enough" (r_cmd="STOP").
# PERP_ROT_MIN/MAX: rotation power limits for the perpendicularity correction.
# PERP_GAIN:    maps skew_px to rotation power.  Start small — this is a fine trim.
PERP_TOL_PX   = 8      # px — skew within ±8px = perpendicular enough
PERP_GAIN     = 0.4    # rotation power per pixel of skew  (8px → 3.2 power units)
PERP_ROT_MIN  = 5      # minimum rotation power when correcting
PERP_ROT_MAX  = 20     # maximum rotation power for perpendicularity correction

# ── Search sweep ──────────────────────────────────────────────────────────────
# When do_ball_align is called with a search_dir, the robot rotates slowly in
# that direction until it sees the pedestal or ball, then begins alignment.
# The camera is rear-facing, so:
#   search_dir="LEFT"  → robot body turns RIGHT (vR = +SEARCH_ROT_POWER)
#   search_dir="RIGHT" → robot body turns LEFT  (vR = -SEARCH_ROT_POWER)
SEARCH_ROT_POWER  = 25    # rotation power during sweep (slow and steady)
SEARCH_TIMEOUT_S  = 16.0   # give up searching after this many seconds

# ── Perpendicularity gate ─────────────────────────────────────────────────────
# When the pedestal blob width in the image exceeds this threshold, the robot is
# getting close enough that a bad angle will cause a miss.  Past this threshold,
# the robot will NOT approach further (y_cmd forced to STOP) until the skew is
# within PERP_TOL_PX — it only rotates and strafes to fix the angle.
# Below this threshold the robot approaches freely (still correcting angle, but
# the approach is allowed even if not yet perfectly perpendicular).
#
# Tune: run ball_align.py standalone, back up to comfortable pickup distance,
# note the blue bbox width_px shown in the debug overlay, set this ~20% less.
PERP_GATE_WIDTH_PX = 350  # px — approaching blocked until perpendicular above this

# Broadcast port for ball_align telemetry
# We reuse TELEMETRY_PORT but add ball_align_* fields so mission_controller
# (which already listens on that port) picks them up automatically.
# Alternatively set to a separate port if you want isolation.
BALL_ALIGN_PORT = 5006   # separate port so it doesn't conflict with new_main3

# Camera resolution for the back camera (use lower for Pi performance)
BACK_CAM_RESOLUTION = (1280, 720)
BACK_CAM_FPS        = 30

# ── Debug image saving ────────────────────────────────────────────────────────
# Set BALL_DEBUG_SAVE_INTERVAL to 0 to disable periodic saves.
BALL_DEBUG_SAVE_DIR      = "ball_debug_images"
BALL_DEBUG_SAVE_INTERVAL = 0.0   # seconds between saves (tune to 0 to disable)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_ball(frame_bgr):
    """
    Find the best white ball candidate in the frame.

    Strategy:
    1. Detect all white blobs (low S, high V) in HSV.
    2. Detect the blue pedestal region.
    3. Only keep white blobs that are ABOVE the blue region, or that exist
       in the upper portion of the frame (when blue isn't fully visible yet).
    4. Of the remaining candidates, pick the LARGEST blob (nearest ball).
    5. Return detection info dict, or None if nothing found.

    Returns dict with:
      centroid_x, centroid_y: pixel coords of ball centre
      area:                   pixel area of white blob
      width_px:               horizontal span of blob bounding box
      height_px:              vertical span of blob bounding box
      frame_w, frame_h:       frame dimensions
      lateral_err_px:         centroid_x - frame_w/2  (+ = ball right of centre)
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # ── White ball mask ───────────────────────────────────────────────────────
    # White = any hue, very low saturation, high value
    white_mask = cv2.inRange(
        hsv,
        np.array([WHITE_H_MIN, WHITE_S_MIN, WHITE_V_MIN]),
        np.array([WHITE_H_MAX, WHITE_S_MAX, WHITE_V_MAX])
    )

    # ── Blue pedestal mask ────────────────────────────────────────────────────
    blue_mask = cv2.inRange(
        hsv,
        np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN]),
        np.array([BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX])
    )

    # ── Restrict search zone ──────────────────────────────────────────────────
    # Search region: top SEARCH_TOP_FRACTION of frame
    search_limit_y = int(h * SEARCH_TOP_FRACTION)
    search_mask = np.zeros((h, w), dtype=np.uint8)
    search_mask[:search_limit_y, :] = 255
    white_mask = cv2.bitwise_and(white_mask, search_mask)

    # ── Pedestal-aware filtering ──────────────────────────────────────────────
    # If blue is visible, only keep white blobs that are ABOVE the blue pedestal.
    # FIX: find the largest blue blob first (ensures it's the real pedestal, not
    # noise), then use that blob's actual bounding-box top edge as the cutoff.
    # Old code used np.argmax() on a row-sum which returns 0 on all-False arrays
    # (no blue found) and is fooled by single noisy pixels in the top rows.
    blue_pixels = cv2.countNonZero(blue_mask)
    pedestal_top_y = None   # will be set if we find a real pedestal blob
    if blue_pixels > MIN_PEDESTAL_AREA_PX // 4:
        # Quick pre-morphology check to see if there's enough blue to bother
        _bk = np.ones((7, 7), np.uint8)
        _blue_clean = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  _bk, iterations=1)
        _blue_clean = cv2.morphologyEx(_blue_clean, cv2.MORPH_CLOSE, _bk, iterations=2)
        _bcs, _ = cv2.findContours(_blue_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if _bcs:
            _best_blue = max(_bcs, key=cv2.contourArea)
            if cv2.contourArea(_best_blue) >= MIN_PEDESTAL_AREA_PX:
                # Use the TOP EDGE of the largest blue blob as the cutoff line.
                # This is the pedestal's actual top boundary in the image.
                _bbx, _bby, _bbw, _bbh = cv2.boundingRect(_best_blue)
                pedestal_top_y = _bby   # y=0 is top; blob starts here
                # Add a small margin so we don't accidentally clip the bottom
                # of the ball when it sits right on top of the pedestal.
                PEDESTAL_TOP_MARGIN_PX = 10
                cutoff_y = max(0, pedestal_top_y - PEDESTAL_TOP_MARGIN_PX)
                above_blue_mask = np.zeros((h, w), dtype=np.uint8)
                above_blue_mask[:cutoff_y, :] = 255
                white_mask = cv2.bitwise_and(white_mask, above_blue_mask)

    # ── Morphology to clean noise ─────────────────────────────────────────────
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Find contours ─────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by minimum area AND circularity.
    # Circularity = 4π·A / P²  (1.0 = perfect circle; floor blobs are << 0.3)
    # When close up, the bottom of the ball is cut off, leaving a half-circle
    # shape (circularity ≈ 0.39), so MIN_BALL_CIRCULARITY is set conservatively.
    def _circularity(c):
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if p < 1:
            return 0.0
        return (4 * math.pi * a) / (p * p)

    # Categorise every contour for the debug overlay:
    #   "too_small"  — failed MIN_BALL_AREA_PX
    #   "not_round"  — passed area, failed circularity threshold
    #   "pass"       — passed both → candidate ball
    #
    # Circularity threshold is adaptive: large blobs (robot close to ball,
    # bottom of ball cut off) use the relaxed CLOSE threshold; small blobs
    # (full ball visible from afar) use the strict FAR threshold.
    debug_contours = []   # list of (contour, category, circularity)
    valid = []
    for c in contours:
        a = cv2.contourArea(c)
        circ = _circularity(c)
        if a < MIN_BALL_AREA_PX:
            debug_contours.append((c, "too_small", circ))
            continue
        circ_thresh = (MIN_BALL_CIRCULARITY_CLOSE
                       if a >= CIRC_SWITCH_AREA_PX
                       else MIN_BALL_CIRCULARITY_FAR)
        if circ < circ_thresh:
            debug_contours.append((c, "not_round", circ))
            continue
        debug_contours.append((c, "pass", circ))
        valid.append(c)

    if not valid:
        return None

    # Pick the largest blob (closest / most prominent ball)
    best = max(valid, key=cv2.contourArea)
    area = cv2.contourArea(best)

    # Centroid
    M = cv2.moments(best)
    if M["m00"] < 1:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Bounding box
    bx, by, bw, bh = cv2.boundingRect(best)

    return {
        "centroid_x":    cx,
        "centroid_y":    cy,
        "area":          area,
        "width_px":      bw,
        "height_px":     bh,
        "bbox":          (bx, by, bw, bh),
        "frame_w":       w,
        "frame_h":       h,
        "lateral_err_px": cx - w // 2,
        "blue_visible":  pedestal_top_y is not None,
        "pedestal_top_y": pedestal_top_y,
        "white_mask":    white_mask,
        "blue_mask":     blue_mask,
        "debug_contours": debug_contours,
    }


def _get_white_debug_contours(frame_bgr):
    """
    Re-run the white-mask pipeline and return ALL contours categorised by
    which filters they pass/fail — used only for the debug mini-mask overlay.

    Returns list of (contour, category, circularity) where category is:
      "too_small"  — area < MIN_BALL_AREA_PX
      "not_round"  — area OK, circularity < MIN_BALL_CIRCULARITY
      "pass"       — accepted as a ball candidate
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(
        hsv,
        np.array([WHITE_H_MIN, WHITE_S_MIN, WHITE_V_MIN]),
        np.array([WHITE_H_MAX, WHITE_S_MAX, WHITE_V_MAX]),
    )
    blue_mask = cv2.inRange(
        hsv,
        np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN]),
        np.array([BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX]),
    )

    # Same search-zone restriction as detect_ball
    search_limit_y = int(h * SEARCH_TOP_FRACTION)
    search_mask = np.zeros((h, w), dtype=np.uint8)
    search_mask[:search_limit_y, :] = 255
    white_mask = cv2.bitwise_and(white_mask, search_mask)

    # Same pedestal cutoff as detect_ball
    blue_pixels = cv2.countNonZero(blue_mask)
    if blue_pixels > MIN_PEDESTAL_AREA_PX // 4:
        _bk = np.ones((7, 7), np.uint8)
        _bc = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  _bk, iterations=1)
        _bc = cv2.morphologyEx(_bc,       cv2.MORPH_CLOSE, _bk, iterations=2)
        _bcs, _ = cv2.findContours(_bc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if _bcs:
            _bb = max(_bcs, key=cv2.contourArea)
            if cv2.contourArea(_bb) >= MIN_PEDESTAL_AREA_PX:
                _, _bby, _, _ = cv2.boundingRect(_bb)
                cutoff_y = max(0, _bby - 10)
                above = np.zeros((h, w), dtype=np.uint8)
                above[:cutoff_y, :] = 255
                white_mask = cv2.bitwise_and(white_mask, above)

    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = []
    for c in contours:
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        circ = (4 * math.pi * a / (p * p)) if p > 1 else 0.0
        if a < MIN_BALL_AREA_PX:
            result.append((c, "too_small", circ))
        else:
            circ_thresh = (MIN_BALL_CIRCULARITY_CLOSE
                           if a >= CIRC_SWITCH_AREA_PX
                           else MIN_BALL_CIRCULARITY_FAR)
            if circ < circ_thresh:
                result.append((c, "not_round", circ))
            else:
                result.append((c, "pass", circ))
    return white_mask, result


def detect_pedestal(frame_bgr):
    """
    Find the blue pedestal in the frame when no white ball is visible.

    Used as a fallback: if the robot can't see the white ball yet (e.g. it's
    off to one side), centring the blue pedestal laterally will slide the robot
    until the white ball appears above it.

    Returns dict with:
      centroid_x, centroid_y: pixel coords of pedestal blob centre
      area:                   pixel area of blue blob
      frame_w, frame_h:       frame dimensions
      lateral_err_px:         centroid_x - frame_w/2  (+ = pedestal right of centre)
    or None if no blue region found.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(
        hsv,
        np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN]),
        np.array([BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX]),
    )

    # Clean noise
    kernel = np.ones((7, 7), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick largest blue blob; it must be big enough to be the real pedestal
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < MIN_PEDESTAL_AREA_PX:   # ignore tiny blue specks
        return None

    M = cv2.moments(best)
    if M["m00"] < 1:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    bx, by, bw, bh = cv2.boundingRect(best)

    return {
        "centroid_x":    cx,
        "centroid_y":    cy,
        "area":          int(area),
        "width_px":      bw,
        "height_px":     bh,
        "bbox":          (bx, by, bw, bh),
        "top_y":         by,   # topmost pixel row of the pedestal blob
        "frame_w":       w,
        "frame_h":       h,
        "lateral_err_px": cx - w // 2,
        "blue_mask":     blue_mask,
    }


def compute_pedestal_skew(blue_mask, bbox):
    """
    Measure how skewed the pedestal's top edge is in the camera image.

    A perfectly perpendicular approach produces a horizontal top edge on the
    pedestal blob.  Any angular misalignment tilts that edge, making one side
    appear higher (smaller y, since y=0 is the top of the frame) than the other.

    Method: split the pedestal bounding box into left and right halves, find
    the topmost non-zero row in each half, then compare.

    Returns:
        skew_px (int):
          = right_top_y - left_top_y
          Positive → right side appears LOWER  → robot angled LEFT  → TURN RIGHT (vR > 0)
          Negative → left  side appears LOWER  → robot angled RIGHT → TURN LEFT  (vR < 0)
          Zero (or very small) → perpendicular ✓
        Returns 0 if the blob is too narrow to split reliably.
    """
    bx, by, bw, bh = bbox
    if bw < 60:   # too narrow to split meaningfully
        return 0

    # Inset slightly from the outer edges to avoid noisy border pixels
    inset = max(4, bw // 10)
    mid   = bx + bw // 2

    left_region  = blue_mask[:, bx + inset : mid]
    right_region = blue_mask[:, mid : bx + bw - inset]

    def _topmost_y(region):
        """Smallest row-index (topmost in image) that contains a non-zero pixel."""
        rows = np.any(region > 0, axis=1)
        idxs = np.where(rows)[0]
        return int(idxs[0]) if len(idxs) > 0 else None

    left_y  = _topmost_y(left_region)
    right_y = _topmost_y(right_region)

    if left_y is None or right_y is None:
        return 0

    return right_y - left_y


def compute_align_state(detection, pedestal=None):
    """
    Given a ball detection (and optionally a pedestal detection as fallback),
    compute the alignment state.

    Priority:
      1. White ball found → align to ball exactly as before (lateral + distance).
      2. No white ball, but blue pedestal visible → align pedestal to centre
         laterally only (no back/forward command).  Distance is never "OK"
         in this mode; we are just sliding until the ball appears.
      3. Nothing found → LOST.

    Returns dict with:
      ball_found:     bool — a white ball is being tracked
      pedestal_mode:  bool — using pedestal as fallback (no ball yet)
      lateral_ok:     bool — horizontally centred enough
      distance_ok:    bool — close enough (always False in pedestal mode)
      aligned:        bool — both lateral and distance OK (False in pedestal mode)
      lateral_err_px: signed pixel error (+ = target right of centre)
      area:           white blob area (0 in pedestal mode)
      distance_state: "FAR" | "CLOSE" | "LOST" | "PEDESTAL"
      x_cmd:          "LEFT" | "RIGHT" | "STOP"
      y_cmd:          "BACK" | "STOP"
      skew_px:        pedestal top-edge skew (+ = right lower = turn right)
      r_cmd:          "LEFT" | "RIGHT" | "STOP"  — perpendicularity rotation
    """
    # ── Perpendicularity from blue pedestal skew ───────────────────────────────
    # Measure from whichever detection has a blue mask available.
    # Prefer the pedestal dict (larger, more stable blob) over the ball detection's
    # smaller blue region at the bottom of the ball image.
    skew_px = 0
    _ped_for_skew = pedestal if pedestal is not None else None
    if _ped_for_skew is None and detection is not None and detection.get("blue_visible"):
        # Fall back to the blue region captured alongside the ball detection
        # (detect_ball already runs the blue mask — reuse it)
        _blue_mask_for_skew = detection.get("blue_mask")
        if _blue_mask_for_skew is not None:
            # Build a rough bbox from the blue mask
            blue_contours, _ = cv2.findContours(
                _blue_mask_for_skew, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if blue_contours:
                best_blue = max(blue_contours, key=cv2.contourArea)
                if cv2.contourArea(best_blue) > 1000:
                    skew_px = compute_pedestal_skew(
                        _blue_mask_for_skew, cv2.boundingRect(best_blue))
    elif _ped_for_skew is not None:
        skew_px = compute_pedestal_skew(
            _ped_for_skew["blue_mask"], _ped_for_skew["bbox"])

    # Rotation command from skew
    if abs(skew_px) <= PERP_TOL_PX:
        r_cmd = "STOP"
    elif skew_px > 0:
        r_cmd = "RIGHT"   # right side lower → robot angled left → turn right
    else:
        r_cmd = "LEFT"

    # ── Pedestal width in pixels ───────────────────────────────────────────────
    # Used by do_ball_align to decide whether to gate the approach.
    # Take from whichever source is available.
    ped_width_px = 0
    if pedestal is not None:
        ped_width_px = pedestal.get("width_px", 0)
    elif detection is not None and detection.get("blue_visible"):
        _bm = detection.get("blue_mask")
        if _bm is not None:
            _bcs, _ = cv2.findContours(_bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if _bcs:
                _bb = max(_bcs, key=cv2.contourArea)
                if cv2.contourArea(_bb) > 500:
                    ped_width_px = cv2.boundingRect(_bb)[2]
    # ── Case 1: white ball visible ─────────────────────────────────────────────
    if detection is not None:
        lat_err = detection["lateral_err_px"]
        area    = detection["area"]

        lateral_ok  = abs(lat_err) < LATERAL_TOL_PX_ALIGN
        distance_ok = area >= AREA_CLOSE_THRESHOLD

        if lat_err > LATERAL_TOL_PX_ALIGN:
            x_cmd = "LEFT"
        elif lat_err < -LATERAL_TOL_PX_ALIGN:
            x_cmd = "RIGHT"
        else:
            x_cmd = "STOP"

        y_cmd = "STOP" if distance_ok else "BACK"

        return {
            "ball_found":    True,
            "pedestal_mode": False,
            "lateral_ok":    lateral_ok,
            "distance_ok":   distance_ok,
            "aligned":       lateral_ok and distance_ok,
            "lateral_err_px": lat_err,
            "area":          int(area),
            "distance_state": "CLOSE" if distance_ok else "FAR",
            "x_cmd":         x_cmd,
            "y_cmd":         y_cmd,
            "skew_px":       skew_px,
            "r_cmd":         r_cmd,
            "pedestal_width_px": ped_width_px,
        }
    if pedestal is not None:
        lat_err = pedestal["lateral_err_px"]
        lateral_ok = abs(lat_err) < PEDESTAL_LATERAL_TOL_PX

        if lat_err > PEDESTAL_LATERAL_TOL_PX:
            x_cmd = "LEFT"
        elif lat_err < -PEDESTAL_LATERAL_TOL_PX:
            x_cmd = "RIGHT"
        else:
            x_cmd = "STOP"

        return {
            "ball_found":    False,
            "pedestal_mode": True,
            "lateral_ok":    lateral_ok,
            "distance_ok":   False,     # never OK without the ball
            "aligned":       False,     # can't be fully aligned without ball
            "lateral_err_px": lat_err,
            "area":          0,
            "distance_state": "PEDESTAL",
            "x_cmd":         x_cmd,
            "y_cmd":         "STOP",    # no forward/back in pedestal mode
            "skew_px":       skew_px,
            "r_cmd":         r_cmd,
            "pedestal_width_px": ped_width_px,
        }

    # ── Case 3: nothing found ──────────────────────────────────────────────────
    return {
        "ball_found":    False,
        "pedestal_mode": False,
        "lateral_ok":    False,
        "distance_ok":   False,
        "aligned":       False,
        "lateral_err_px": None,
        "area":          0,
        "distance_state": "LOST",
        "x_cmd":         "STOP",
        "y_cmd":         "STOP",
        "skew_px":       0,
        "r_cmd":         "STOP",
        "pedestal_width_px": 0,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEMPORAL STABILITY — flicker suppression
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectionStabilizer:
    """
    Wraps detect_ball / detect_pedestal + compute_align_state with temporal
    smoothing to eliminate single-frame flicker.

    Two independent hysteresis counters:

    1. Detection presence (ball / pedestal / lost):
       - A NEW source (e.g. LOST→BALL) must appear DETECT_CONFIRM_FRAMES times
         consecutively before it becomes the accepted state.
       - An EXISTING source must DISAPPEAR for DETECT_LOSS_FRAMES consecutive
         frames before we switch away from it (hysteresis keeps the last good
         state alive briefly so small occlusions don't cause jitter).

    2. Command / state values (x_cmd, distance_state):
       - The most-recently-emitted value is kept until a DIFFERENT value has
         appeared STATE_CONFIRM_FRAMES times in a row.  This prevents rapid
         LEFT/RIGHT or FAR/CLOSE switching caused by a centroid hovering near
         the tolerance boundary.

    3. Pedestal-seen latch (anti-false-floor-detection):
       - Ball detections are IGNORED until the pedestal has been confirmed at
         least once.  Once the pedestal is seen, the latch latches permanently
         for this run — so when the robot gets so close that the pedestal
         disappears behind the ball, ball tracking continues uninterrupted.
       - Reset via reset_pedestal_latch() if you need to re-run from scratch.
    """

    def __init__(self):
        # ── Detection source hysteresis ────────────────────────────────────────
        # accepted source: "ball" | "pedestal" | "lost"
        self._accepted_src  = "lost"
        self._candidate_src = "lost"
        self._candidate_cnt = 0          # consecutive frames the candidate has held
        self._loss_cnt      = 0          # consecutive frames current source absent

        # Last accepted raw results (used while waiting for confirmation)
        self._last_detection = None
        self._last_pedestal  = None

        # ── Pedestal-seen latch ────────────────────────────────────────────────
        # Flips True the first time a pedestal is confirmed; never resets unless
        # reset_pedestal_latch() is called explicitly.  Ball detections are
        # suppressed (treated as "lost") until this is True.
        self._pedestal_seen = False

        # ── Command stabilisation ──────────────────────────────────────────────
        # For each key we track: current emitted value, candidate value, count
        self._cmd_state = {}   # key → {"current": v, "candidate": v, "cnt": n}

    def reset_pedestal_latch(self):
        """Call this to re-arm the latch (e.g. when starting a new pickup)."""
        self._pedestal_seen = False

    def _confirm_cmd(self, key: str, new_val):
        """Return the stable value for a command key, applying hysteresis."""
        if key not in self._cmd_state:
            self._cmd_state[key] = {"current": new_val, "candidate": new_val, "cnt": STATE_CONFIRM_FRAMES}
        s = self._cmd_state[key]
        if new_val == s["current"]:
            # Same as current — reset candidate tracking
            s["candidate"] = new_val
            s["cnt"] = 0
            return new_val
        # Different from current
        if new_val == s["candidate"]:
            s["cnt"] += 1
        else:
            s["candidate"] = new_val
            s["cnt"] = 1
        if s["cnt"] >= STATE_CONFIRM_FRAMES:
            s["current"]   = new_val
            s["candidate"] = new_val
            s["cnt"]       = 0
        return s["current"]

    def update(self, frame_bgr):
        """
        Run detection on a frame and return a temporally-stable
        (detection, pedestal, state) tuple — same types as the raw functions.
        """
        raw_det = detect_ball(frame_bgr)
        raw_ped = detect_pedestal(frame_bgr) if raw_det is None else None

        # ── Pedestal-seen gate ─────────────────────────────────────────────────
        # Suppress ball detections entirely until we have confirmed the pedestal
        # at least once.  This prevents the white floor being mistaken for a ball
        # before the robot has even found the pedestal.
        # Once latched, ball detection is always permitted — even if the pedestal
        # is no longer visible (because the ball fills the frame up close).
        if not self._pedestal_seen and raw_det is not None:
            raw_det = None   # ignore — pedestal not confirmed yet

        # ── Determine raw source this frame ────────────────────────────────────
        if raw_det is not None:
            raw_src = "ball"
        elif raw_ped is not None:
            raw_src = "pedestal"
        else:
            raw_src = "lost"

        # ── Detection source hysteresis ────────────────────────────────────────
        if raw_src == self._accepted_src:
            # Continuing with the same source — update last known good data
            if raw_src == "ball":
                self._last_detection = raw_det
            elif raw_src == "pedestal":
                self._last_pedestal  = raw_ped
            self._loss_cnt      = 0
            self._candidate_src = raw_src
            self._candidate_cnt = 0
        else:
            # Source has changed from accepted
            self._loss_cnt += 1

            # Track consecutive frames of the new candidate
            if raw_src == self._candidate_src:
                self._candidate_cnt += 1
            else:
                self._candidate_src = raw_src
                self._candidate_cnt = 1

            # Decide whether to switch
            if raw_src == "lost":
                # Require DETECT_LOSS_FRAMES to confirm a loss
                if self._loss_cnt >= DETECT_LOSS_FRAMES:
                    self._accepted_src   = "lost"
                    self._last_detection = None
                    self._last_pedestal  = None
            else:
                # Require DETECT_CONFIRM_FRAMES to confirm a new detection
                if self._candidate_cnt >= DETECT_CONFIRM_FRAMES:
                    self._accepted_src = raw_src
                    self._loss_cnt     = 0
                    if raw_src == "ball":
                        self._last_detection = raw_det
                        self._last_pedestal  = None
                    else:
                        # Pedestal just confirmed — set the latch permanently
                        self._pedestal_seen  = True
                        self._last_pedestal  = raw_ped
                        self._last_detection = None

        # ── Compute raw state from accepted source ─────────────────────────────
        det_to_use = self._last_detection if self._accepted_src == "ball"     else None
        ped_to_use = self._last_pedestal  if self._accepted_src == "pedestal" else None
        raw_state  = compute_align_state(det_to_use, ped_to_use)

        # ── Stabilise individual commands ──────────────────────────────────────
        stable_state = dict(raw_state)
        for key in ("x_cmd", "y_cmd", "r_cmd", "distance_state"):
            stable_state[key] = self._confirm_cmd(key, raw_state[key])

        # Re-derive aligned from stabilised commands (so it doesn't flicker either)
        if stable_state["ball_found"]:
            stable_state["aligned"] = (
                stable_state["lateral_ok"] and
                stable_state["distance_state"] == "CLOSE"
            )

        # Expose latch state so telemetry / debug overlay can show it
        stable_state["pedestal_seen"] = self._pedestal_seen

        return det_to_use, ped_to_use, stable_state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TELEMETRY BROADCAST  (mission_controller subscribes to BALL_ALIGN_PORT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BallAlignBroadcaster:
    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, state: dict):
        try:
            self._sock.sendto(
                json.dumps(state).encode(),
                ("127.0.0.1", BALL_ALIGN_PORT)
            )
        except Exception:
            pass

    def close(self):
        self._sock.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATE SUBSCRIBER  (used by mission_controller)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BallAlignState:
    """
    Subscribes to ball_align.py's telemetry on BALL_ALIGN_PORT.
    Import into mission_controller and use like:

        align = BallAlignState()
        while not align.aligned:
            ...
        align.stop()

    Properties:
      .aligned          bool  — both lateral and distance OK
      .lateral_ok       bool
      .distance_ok      bool
      .lateral_err_px   float | None  — signed error, + = ball right of centre
      .area             int   — white blob area
      .distance_state   str   — "FAR" | "CLOSE" | "LOST"
      .x_cmd            str   — "LEFT" | "RIGHT" | "STOP"
      .y_cmd            str   — "BACK" | "STOP"
      .ball_found       bool
    """
    def __init__(self, port: int = BALL_ALIGN_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.1)
        self._data: dict = {}
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while self._running:
            try:
                raw, _ = self._sock.recvfrom(4096)
                data = json.loads(raw.decode())
                with self._lock:
                    self._data = data
            except (socket.timeout, json.JSONDecodeError):
                pass

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    @property
    def aligned(self) -> bool:
        return self.get().get("aligned", False)

    @property
    def lateral_ok(self) -> bool:
        return self.get().get("lateral_ok", False)

    @property
    def distance_ok(self) -> bool:
        return self.get().get("distance_ok", False)

    @property
    def lateral_err_px(self):
        return self.get().get("lateral_err_px")

    @property
    def area(self) -> int:
        return self.get().get("area", 0)

    @property
    def distance_state(self) -> str:
        return self.get().get("distance_state", "LOST")

    @property
    def x_cmd(self) -> str:
        return self.get().get("x_cmd", "STOP")

    @property
    def y_cmd(self) -> str:
        return self.get().get("y_cmd", "STOP")

    @property
    def ball_found(self) -> bool:
        return self.get().get("ball_found", False)

    @property
    def pedestal_mode(self) -> bool:
        """True when aligning to the blue pedestal as a fallback (no ball yet)."""
        return self.get().get("pedestal_mode", False)

    @property
    def skew_px(self) -> int:
        """Pedestal top-edge skew. Positive = right side lower = robot angled left."""
        return self.get().get("skew_px", 0)

    @property
    def r_cmd(self) -> str:
        """Perpendicularity rotation command: 'LEFT' | 'RIGHT' | 'STOP'."""
        return self.get().get("r_cmd", "STOP")

    def stop(self):
        self._running = False
        try: self._sock.close()
        except: pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MISSION CONTROLLER HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def do_ball_align(mc, align_state: BallAlignState,
                  timeout: float = 30.0,
                  search_dir: str = None,
                  verbose: bool = True) -> bool:
    """
    Drive the robot until the back camera is aligned with a white ball.
    ball_align.py must be running separately (it opens the back camera).

    Args:
        mc:          MovementClient instance
        align_state: BallAlignState instance (already listening)
        timeout:     total seconds before giving up
        search_dir:  "LEFT" | "RIGHT" | None
                     If the robot cannot see the ball/pedestal when called,
                     it rotates in this direction until it finds something,
                     then begins alignment.
                     "LEFT"  → rear camera looks left → robot body turns RIGHT (vR > 0)
                     "RIGHT" → rear camera looks right → robot body turns LEFT  (vR < 0)
                     None    → skip search, start alignment immediately
        verbose:     print status

    Phases:
        SEARCH  — rotate slowly in search_dir until pedestal/ball appears.
        ALIGN   — strafe + perpendicularity-correct + approach simultaneously.
                  Approach (y_cmd=BACK) is GATED when the pedestal is wide
                  (robot is close) but the angle is still off:
                    if pedestal_width_px >= PERP_GATE_WIDTH_PX AND |skew| > PERP_TOL_PX
                    → hold distance, only rotate/strafe until perpendicular.

    Returns True if aligned, False if timed out.
    """
    from movement_client import MoveCmd

    if verbose:
        print(f"[ball_align] Starting  search_dir={search_dir}")

    t0        = time.time()
    stable    = 0
    loss_time = None
    phase     = "SEARCH" if search_dir else "ALIGN"

    while time.time() - t0 < timeout:
        st = align_state.get()
        if not st:
            time.sleep(0.05)
            continue

        ball_found = st.get("ball_found", False)
        ped_mode   = st.get("pedestal_mode", False)
        something  = ball_found or ped_mode
        x_cmd      = st.get("x_cmd",  "STOP")
        y_cmd      = st.get("y_cmd",  "STOP")
        r_cmd      = st.get("r_cmd",  "STOP")
        lat        = st.get("lateral_err_px") or 0
        skew       = st.get("skew_px", 0)
        area       = st.get("area", 0)
        ped_w      = st.get("pedestal_width_px", 0)
        dist_err   = max(0, AREA_CLOSE_THRESHOLD - area)

        if verbose:
            src = "BALL" if ball_found else ("PED" if ped_mode else "LOST")
            print(f"  [{phase}][{src}] "
                  f"lat={lat:+.0f}px  skew={skew:+d}px  "
                  f"ped_w={ped_w}px  area={area}  "
                  f"x={x_cmd} y={y_cmd} r={r_cmd}  "
                  f"{'✓' if st.get('aligned') else ''}")

        # ── PHASE: SEARCH ──────────────────────────────────────────────────────
        # Rotate slowly until the pedestal or ball comes into view, then switch.
        if phase == "SEARCH":
            if something:
                if verbose:
                    print("[ball_align] Target found — switching to ALIGN")
                phase = "ALIGN"
                # Fall through into ALIGN logic this same iteration
            else:
                # Nothing visible yet — keep rotating toward search_dir
                # "LEFT" = want camera to look left = robot body turns RIGHT = vR > 0
                vR = SEARCH_ROT_POWER if search_dir == "LEFT" else -SEARCH_ROT_POWER
                mc.send(MoveCmd(cmd="omni", vX=0, vY=0, vR=vR))
                time.sleep(0.05)
                continue

        # ── PHASE: ALIGN ───────────────────────────────────────────────────────
        if not something:
            # Lost sight during alignment — stop and wait briefly
            loss_time = time.time()
            mc.brake()
            time.sleep(0.05)
            continue

        # Aligned check with stability counter
        if st.get("aligned"):
            if loss_time and time.time() - loss_time < 1.0:
                stable = 0   # reject: just recovered from a loss
            elif time.time() - t0 < 0.75:
                stable = 0   # reject: too soon after start
            else:
                stable += 1
                if stable >= 5:
                    mc.brake()
                    if verbose:
                        print("[ball_align] Aligned!")
                    return True
        else:
            stable = 0

        # ── vX: lateral strafe ─────────────────────────────────────────────────
        strafe_base = ALIGN_PEDESTAL_STRAFE_POWER if ped_mode else ALIGN_STRAFE_POWER
        vX = {
            "RIGHT":  strafe_base * abs(lat) / 300,
            "LEFT":  -strafe_base * abs(lat) / 300,
        }.get(x_cmd, 0)

        # ── vY: approach — gated by perpendicularity when close ────────────────
        # If the pedestal blob is already wide (robot is close) AND the robot is
        # still angled, stop advancing.  Better to fix the angle now than to
        # arrive at pickup distance misaligned with no room to correct.
        perp_ok   = abs(skew) <= PERP_TOL_PX
        gate_hit  = (ped_w >= PERP_GATE_WIDTH_PX and not perp_ok)
        if gate_hit:
            vY = 0
            if verbose and y_cmd == "BACK":
                print(f"  [PERP GATE] ped_w={ped_w}>={PERP_GATE_WIDTH_PX}, "
                      f"skew={skew:+d} — holding until perpendicular")
        elif y_cmd == "BACK":
            scale = min(dist_err / AREA_CLOSE_THRESHOLD, 1.0)
            vY = -ALIGN_BACK_POWER * scale
        else:
            vY = 0

        # ── vR: perpendicularity trim ──────────────────────────────────────────
        # skew > 0 → right side lower → robot angled left → TURN RIGHT (vR > 0)
        # skew < 0 → left  side lower → robot angled right → TURN LEFT  (vR < 0)
        if r_cmd != "STOP":
            rot_raw = abs(skew) * PERP_GAIN
            rot_spd = max(PERP_ROT_MIN, min(PERP_ROT_MAX, rot_raw))
            vR = rot_spd if skew > 0 else -rot_spd
        else:
            vR = 0

        mc.send(MoveCmd(cmd="omni", vX=vX, vY=vY, vR=vR))
        time.sleep(0.05)

    mc.brake()
    if verbose:
        print("[ball_align] TIMEOUT — alignment failed.")
    return False
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN — standalone camera loop with live debug display
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _next_save_index(folder: str) -> int:
    """Return the next available numeric index in folder (existing max + 1)."""
    os.makedirs(folder, exist_ok=True)
    existing = []
    for f in os.listdir(folder):
        name = os.path.splitext(f)[0]   # strip extension
        for prefix in ("clean_", "debug_"):
            if name.startswith(prefix):
                try: existing.append(int(name[len(prefix):]))
                except ValueError: pass
    return max(existing, default=0) + 1


def main():
    print("Ball Align — press Q to quit, T to toggle debug overlay")
    print(f"Broadcasting on UDP port {BALL_ALIGN_PORT}")
    print(f"Tuning constants:")
    print(f"  AREA_CLOSE_THRESHOLD = {AREA_CLOSE_THRESHOLD}  (increase = must be closer)")
    print(f"  LATERAL_TOL_PX_ALIGN = {LATERAL_TOL_PX_ALIGN}  (decrease = stricter centering)")
    print()

    # Open back camera
    if sys.platform.startswith('win'):
        cap = cv2.VideoCapture(CAM_BACK, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAM_BACK, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"ERROR: Cannot open back camera '{CAM_BACK}'")
        print("Check config.py CAM_BACK and that the device exists.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  BACK_CAM_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, BACK_CAM_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, BACK_CAM_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w == 0 or actual_h == 0:
        print(f"ERROR: Camera returned 0x0 resolution.")
        cap.release()
        return
    print(f"Camera: {actual_w}x{actual_h}")

    for _ in range(5): cap.read()   # warm up

    broadcaster = BallAlignBroadcaster()
    stabilizer  = DetectionStabilizer()
    show_debug  = True
    frame_count = 0
    last_save_t = 0.0
    save_idx    = _next_save_index(BALL_DEBUG_SAVE_DIR)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        detection, pedestal, state = stabilizer.update(frame)
        broadcaster.send(state)

        # ── Console output every 10 frames ───────────────────────────────────
        if frame_count % 10 == 0:
            if state["ball_found"]:
                print(f"  [BALL]     lat={state['lateral_err_px']:+4.0f}px  "
                      f"area={state['area']:7d}  "
                      f"dist={state['distance_state']:5}  "
                      f"x={state['x_cmd']:5}  y={state['y_cmd']:4}  "
                      f"{'✓ ALIGNED' if state['aligned'] else ''}")
            elif state["pedestal_mode"]:
                print(f"  [PEDESTAL] lat={state['lateral_err_px']:+4.0f}px  "
                      f"x={state['x_cmd']:5}  (waiting for white ball)")
            else:
                print("  [LOST]     nothing detected")

        # ── Debug display ─────────────────────────────────────────────────────
        if show_debug:
            display = frame.copy()
            cx_frame = actual_w // 2

            # Draw frame centre line
            cv2.line(display, (cx_frame, 0), (cx_frame, actual_h),
                     (0, 255, 255), 1)

            # Draw search zone limit
            limit_y = int(actual_h * SEARCH_TOP_FRACTION)
            cv2.line(display, (0, limit_y), (actual_w, limit_y),
                     (128, 0, 128), 1)

            if detection is not None:
                bx, by, bw, bh = detection["bbox"]
                # Ball bounding box
                color = (0, 255, 0) if state["aligned"] else \
                        (0, 255, 255) if state["lateral_ok"] else (0, 120, 255)
                cv2.rectangle(display, (bx, by), (bx+bw, by+bh), color, 2)

                # Centroid dot
                cx, cy = detection["centroid_x"], detection["centroid_y"]
                cv2.circle(display, (cx, cy), 6, color, -1)

                # Lateral error line
                cv2.line(display, (cx_frame, cy), (cx, cy), (0, 200, 255), 2)

                # Stats overlay
                lat  = state["lateral_err_px"]
                area = state["area"]
                dst  = state["distance_state"]
                xcmd = state["x_cmd"]
                ycmd = state["y_cmd"]
                skew = state["skew_px"]
                rcmd = state["r_cmd"]
                txt = (f"BALL  lat={lat:+.0f}px  {xcmd} | "
                       f"area={area//1000}k  dist={dst}  {ycmd} | "
                       f"skew={skew:+d}px  {rcmd}")
                cv2.putText(display, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if state["aligned"]:
                    cv2.putText(display, "ALIGNED", (cx_frame - 60, actual_h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                # Draw a skew indicator: horizontal line through centroid
                # tilted by skew amount to show the angle visually
                if abs(skew) > 0:
                    skew_col = (0, 0, 255) if rcmd != "STOP" else (0, 200, 0)
                    cv2.putText(display, f"PERP: {rcmd} ({skew:+d}px)",
                                (10, actual_h - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, skew_col, 2)

            elif pedestal is not None:
                # Pedestal fallback mode — draw pedestal in blue/orange
                bx, by, bw, bh = pedestal["bbox"]
                ped_color = (255, 160, 0) if state["lateral_ok"] else (0, 120, 255)
                cv2.rectangle(display, (bx, by), (bx+bw, by+bh), ped_color, 2)

                cx, cy = pedestal["centroid_x"], pedestal["centroid_y"]
                cv2.circle(display, (cx, cy), 8, ped_color, -1)
                cv2.circle(display, (cx, cy), 8, (255, 255, 255), 1)

                # Lateral error line from frame centre to pedestal centroid
                cv2.line(display, (cx_frame, cy), (cx, cy), (255, 200, 50), 2)

                lat  = state["lateral_err_px"]
                xcmd = state["x_cmd"]
                skew = state["skew_px"]
                rcmd = state["r_cmd"]
                txt = (f"PEDESTAL (no ball)  lat={lat:+.0f}px  {xcmd}  "
                       f"skew={skew:+d}px  {rcmd}  | waiting for white ball")
                cv2.putText(display, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, ped_color, 2)

            else:
                cv2.putText(display, "LOST — no ball or pedestal", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ── Pedestal-seen latch indicator (always visible) ────────────────
            _ped_seen = state.get("pedestal_seen", False)
            _latch_txt = "PEDESTAL SEEN [unlocked]" if _ped_seen else "WAITING FOR PEDESTAL [locked]"
            _latch_col = (0, 220, 80) if _ped_seen else (0, 80, 220)
            cv2.putText(display, _latch_txt, (10, actual_h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _latch_col, 2)

            # ── Mini mask panel — colour-coded by filter result ───────────────
            # Always run the debug contour pass so we can show rejected blobs too.
            # Colours (BGR):
            #   GREEN  (0,220,0)    — passed area + roundness → accepted ball
            #   ORANGE (0,140,255)  — passed area, FAILED roundness (too flat)
            #   GREY   (120,120,120)— failed area (too small)
            #   BLUE   (255,60,0)   — blue pedestal mask (right half)
            _dbg_white_mask, _dbg_contours = _get_white_debug_contours(frame)
            dbg_h, dbg_w = _dbg_white_mask.shape[:2]

            # Left panel: colour-coded white blobs on black background
            left_panel = np.zeros((dbg_h, dbg_w, 3), dtype=np.uint8)
            _cat_color = {
                "too_small": (120, 120, 120),   # grey
                "not_round": (0,   140, 255),   # orange
                "pass":      (0,   220,   0),   # green
            }
            for _c, _cat, _circ in _dbg_contours:
                col = _cat_color[_cat]
                cv2.drawContours(left_panel, [_c], -1, col, -1)   # filled
                # Label each blob with its circularity value
                _M = cv2.moments(_c)
                if _M["m00"] > 0:
                    _lx = int(_M["m10"] / _M["m00"])
                    _ly = int(_M["m01"] / _M["m00"])
                    cv2.putText(left_panel, f"{_circ:.2f}",
                                (_lx - 18, _ly + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                                cv2.LINE_AA)

            # Right panel: blue pedestal mask
            right_panel = np.zeros((dbg_h, dbg_w, 3), dtype=np.uint8)
            _raw_blue = cv2.inRange(
                cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN]),
                np.array([BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX]),
            )
            right_panel[_raw_blue > 0] = (255, 60, 0)   # BGR blue-ish

            # Draw pedestal cutoff line on both panels if applicable
            if detection is not None and detection.get("pedestal_top_y") is not None:
                _cy_line = detection["pedestal_top_y"] - 10
                cv2.line(left_panel,  (0, _cy_line), (dbg_w, _cy_line), (0, 255, 255), 1)
                cv2.line(right_panel, (0, _cy_line), (dbg_w, _cy_line), (0, 255, 255), 1)

            side = np.hstack([left_panel, right_panel])
            scale = 0.3
            side_small = cv2.resize(side, (int(side.shape[1] * scale),
                                           int(side.shape[0] * scale)),
                                    interpolation=cv2.INTER_AREA)

            # Legend (drawn on the small panel after resize for legibility)
            # Also show which circularity mode is active for the largest blob.
            _lx0, _ly0 = 4, 10
            for _label, _col in [("pass", (0,220,0)), ("not_round",(0,140,255)), ("too_small",(120,120,120))]:
                cv2.putText(side_small, _label, (_lx0, _ly0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, _col, 1, cv2.LINE_AA)
                _ly0 += 12

            # Show active circularity mode + threshold
            _ball_area = detection["area"] if detection is not None else 0
            _circ_mode = "CLOSE" if _ball_area >= CIRC_SWITCH_AREA_PX else "FAR"
            _circ_thr  = (MIN_BALL_CIRCULARITY_CLOSE
                          if _ball_area >= CIRC_SWITCH_AREA_PX
                          else MIN_BALL_CIRCULARITY_FAR)
            _mode_col  = (0, 200, 255) if _circ_mode == "CLOSE" else (200, 200, 0)
            cv2.putText(side_small, f"circ:{_circ_mode}>={_circ_thr:.2f}",
                        (_lx0, _ly0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, _mode_col, 1, cv2.LINE_AA)

            display[:side_small.shape[0], :side_small.shape[1]] = side_small

            # ── Periodic debug image save ─────────────────────────────────────
            now_t = time.time()
            if BALL_DEBUG_SAVE_INTERVAL > 0 and now_t - last_save_t >= BALL_DEBUG_SAVE_INTERVAL:
                os.makedirs(BALL_DEBUG_SAVE_DIR, exist_ok=True)
                clean_path = os.path.join(BALL_DEBUG_SAVE_DIR, f"clean_{save_idx}.jpg")
                debug_path = os.path.join(BALL_DEBUG_SAVE_DIR, f"debug_{save_idx}.jpg")
                cv2.imwrite(clean_path, frame)
                cv2.imwrite(debug_path, display)
                print(f"  [save] ball_debug #{save_idx} → {BALL_DEBUG_SAVE_DIR}/")
                save_idx  += 1
                last_save_t = now_t

            if HAS_DISPLAY:
                win_w = int(960 * DISPLAY_SCALE)
                win_h = int(540 * DISPLAY_SCALE)

                display_scaled = cv2.resize(
                    display,
                    (win_w, win_h),
                    interpolation=cv2.INTER_AREA
                )

                cv2.namedWindow("Ball Align", cv2.WINDOW_NORMAL)
                cv2.imshow("Ball Align", display_scaled)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    show_debug = not show_debug

    cap.release()
    broadcaster.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()