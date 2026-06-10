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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TUNABLE PARAMETERS
# All values calibrated for the camera view shown in the reference images.
# Tune on the actual robot by running standalone: uv run ball_align.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# White ball HSV thresholds.
# White = very low saturation, very high brightness.
BALL_WHITE_S_MAX  = 70    # max saturation to count as white
BALL_WHITE_V_MIN  = 150   # min brightness to count as white

# Blue podium HSV thresholds.
# Used to reject white blobs that are NOT above blue (e.g. white floor).
BLUE_H_MIN, BLUE_H_MAX = 95,  135
BLUE_S_MIN, BLUE_S_MAX = 100, 255
BLUE_V_MIN, BLUE_V_MAX = 60,  230

# Fraction of frame height to search for balls.
# Ignore the bottom portion to avoid seeing the white floor.
# Balls sit on the blue pedestal which occupies the lower half,
# but the BALL itself is in the upper portion.
# Setting 0.80 means we look in the top 80% of frame for white blobs.
SEARCH_TOP_FRACTION = 1.0

# Minimum blob area in pixels (at the camera's capture resolution).
# Avoids picking up small reflections or noise.
MIN_BALL_AREA_PX = 3000

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

# Broadcast port for ball_align telemetry
# We reuse TELEMETRY_PORT but add ball_align_* fields so mission_controller
# (which already listens on that port) picks them up automatically.
# Alternatively set to a separate port if you want isolation.
BALL_ALIGN_PORT = 5006   # separate port so it doesn't conflict with new_main3

# Camera resolution for the back camera (use lower for Pi performance)
BACK_CAM_RESOLUTION = (1280, 720)
BACK_CAM_FPS        = 30


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
        np.array([0,   0,             BALL_WHITE_V_MIN]),
        np.array([180, BALL_WHITE_S_MAX, 255])
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
    # If blue is visible, only keep white blobs that are ABOVE the blue.
    blue_pixels = cv2.countNonZero(blue_mask)
    if blue_pixels > 500:
        # Find the topmost row that has significant blue
        blue_col_sum = np.sum(blue_mask, axis=1)
        blue_top_row = int(np.argmax(blue_col_sum > 20))
        if blue_top_row > 0:
            # Restrict white search to above the blue pedestal
            above_blue_mask = np.zeros((h, w), dtype=np.uint8)
            above_blue_mask[:blue_top_row, :] = 255
            white_mask = cv2.bitwise_and(white_mask, above_blue_mask)

    # ── Morphology to clean noise ─────────────────────────────────────────────
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Find contours ─────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by minimum area
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_BALL_AREA_PX]
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
        "blue_visible":  blue_pixels > 500,
        "white_mask":    white_mask,
        "blue_mask":     blue_mask,
    }


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

    # Pick largest blue blob
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < 2000:   # ignore tiny blue specks
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
        "frame_w":       w,
        "frame_h":       h,
        "lateral_err_px": cx - w // 2,
        "blue_mask":     blue_mask,
    }


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
    """
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
        }

    # ── Case 2: no ball, pedestal visible → lateral-only fallback ─────────────
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
    }


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

    def stop(self):
        self._running = False
        try: self._sock.close()
        except: pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MISSION CONTROLLER HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def do_ball_align(mc, align_state: BallAlignState,
                  timeout: float = 30.0, verbose: bool = True) -> bool:
    """
    Drive the robot until the back camera is aligned with a white ball.
    ball_align.py must be running separately (it opens the back camera).

    This function is called from mission_controller AFTER you've driven
    the robot to roughly the right position. It does fine lateral + distance
    adjustment using back camera feedback.

    Args:
        mc:          MovementClient instance
        align_state: BallAlignState instance (already listening)
        timeout:     seconds before giving up
        verbose:     print status

    Returns True if aligned, False if timed out.

    Usage:
        # Import at top of mission_controller.py:
        from ball_align import BallAlignState, do_ball_align

        # In your game loop:
        align = BallAlignState()
        success = do_ball_align(mc, align, timeout=20.0)
        align.stop()
        if success:
            mc.action("open_claw")
            mc.omni(vY=-20)   # push into ball
            time.sleep(0.5)
            mc.brake()
            mc.action("close_claw")
    """
    # Import here to avoid circular import issues
    from movement_client import MoveCmd

    if verbose:
        print("[ball_align] Starting alignment...")

    stable = 0
    t0 = time.time()
    loss_time = None
    while time.time() - t0 < timeout:
        st = align_state.get()

        if not st:
            time.sleep(0.05)
            continue

        if verbose:
            bf        = st.get("ball_found", False)
            ped_mode  = st.get("pedestal_mode", False)
            lat       = st.get("lateral_err_px")
            dst       = st.get("distance_state", "?")
            area      = st.get("area", 0)
            dist_err  = max(0, AREA_CLOSE_THRESHOLD - area)
            source    = "BALL" if bf else ("PEDESTAL" if ped_mode else "LOST")
            print(f"  [{source}]  "
                  f"lat={'---' if lat is None else f'{lat:+.0f}px'}  "
                  f"dist={dst}  area={area}  "
                  f"x={st.get('x_cmd','?')}  y={st.get('y_cmd','?')}  "
                  f"{'✓ ALIGNED' if st.get('aligned') else ''}")

        if st.get("aligned"):
            if loss_time and time.time() - loss_time < 1.0:
                print("[ball_align] Alignment After Loss Rejected!")
                stable = 0
            elif time.time() - t0 < 0.75:
                print("[ball_align] Alignment After Start Rejected!")
                stable = 0
            else:
                stable += 1
                if stable >= 5:
                    mc.brake()
                    if verbose: print("[ball_align] Aligned!")
                    return True
        else:
            stable = 0

        x_cmd    = st.get("x_cmd", "STOP")
        y_cmd    = st.get("y_cmd", "STOP")
        ped_mode = st.get("pedestal_mode", False)
        lat      = st.get("lateral_err_px") or 0
        area     = st.get("area", 0)
        dist_err = max(0, AREA_CLOSE_THRESHOLD - area)

        if not st.get("ball_found") and not ped_mode:
            # Completely lost — stop and wait
            loss_time = time.time()
            mc.brake()
            time.sleep(0.05)
            continue

        # ── Lateral strafe ─────────────────────────────────────────────────────
        # Use gentler power when only the pedestal is visible (coarser target).
        strafe_base = ALIGN_PEDESTAL_STRAFE_POWER if ped_mode else ALIGN_STRAFE_POWER
        vX = {
            "RIGHT":  strafe_base * abs(lat) / 300,
            "LEFT":  -strafe_base * abs(lat) / 300,
        }.get(x_cmd, 0)

        # ── Back movement ──────────────────────────────────────────────────────
        # Never commanded in pedestal mode (y_cmd is always "STOP" there).
        if y_cmd == "BACK":
            scale = min((dist_err / AREA_CLOSE_THRESHOLD), 1.0)
            vY = -ALIGN_BACK_POWER * scale
        else:
            vY = 0

        mc.send(MoveCmd(cmd="omni", vX=vX, vY=vY, vR=0))
        time.sleep(0.05)

    mc.brake()
    if verbose: print("[ball_align] TIMEOUT — alignment failed.")
    return False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN — standalone camera loop with live debug display
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    show_debug  = True
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        detection = detect_ball(frame)
        # Pedestal fallback: only run if no white ball found (saves CPU)
        pedestal  = detect_pedestal(frame) if detection is None else None
        state     = compute_align_state(detection, pedestal)
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
                txt = (f"BALL  lat={lat:+.0f}px  {xcmd} | "
                       f"area={area//1000}k  dist={dst}  {ycmd}")
                cv2.putText(display, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if state["aligned"]:
                    cv2.putText(display, "ALIGNED", (cx_frame - 60, actual_h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

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
                txt = f"PEDESTAL (no ball)  lat={lat:+.0f}px  {xcmd}  | waiting for white ball"
                cv2.putText(display, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, ped_color, 2)

            else:
                cv2.putText(display, "LOST — no ball or pedestal", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Show masks side by side at reduced size
            if detection is not None:
                wm = cv2.cvtColor(detection["white_mask"], cv2.COLOR_GRAY2BGR)
                bm = cv2.cvtColor(detection["blue_mask"],  cv2.COLOR_GRAY2BGR)
                side = np.hstack([wm, bm])
                scale = 0.3
                side_small = cv2.resize(side, (int(side.shape[1]*scale),
                                                int(side.shape[0]*scale)))
                display[:side_small.shape[0], :side_small.shape[1]] = side_small
            elif pedestal is not None:
                bm = cv2.cvtColor(pedestal["blue_mask"], cv2.COLOR_GRAY2BGR)
                scale = 0.3
                bm_small = cv2.resize(bm, (int(bm.shape[1]*scale),
                                           int(bm.shape[0]*scale)))
                display[:bm_small.shape[0], :bm_small.shape[1]] = bm_small

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