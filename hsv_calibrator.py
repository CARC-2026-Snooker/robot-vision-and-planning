"""
hsv_calibrator.py — Interactive HSV range calculator for camera colour detection.

Usage:
    python hsv_calibrator.py [image1.jpg image2.jpg ...]
    (no args = file dialog)

Controls:
    Left-click          → place pin at current label
    Right-click         → remove nearest pin
    Scroll              → zoom (sampling radius adapts automatically)
    Middle-drag         → pan
    1–9                 → select label
    A                   → add label (type name in terminal)
    D                   → delete active label + its pins
    X                   → toggle separation pair (active label ↔ another)
    ← →                 → adjust trim percentile  (default 10 = keep middle 80%)
    N                   → load more images / cycle images
    ENTER               → compute HSV ranges
    S                   → save results to hsv_results.json
    Q / Esc             → quit

Sampling is zoom-adaptive: at low zoom a large neighbourhood is averaged per pin
(robust against noise), at high zoom it narrows to individual pixels.

Computation:
    1. Percentile trim (p_lo … p_hi) per channel — rejects outliers / speculars.
    2. Hue wrap detection — red objects that straddle H=0/180 automatically
       produce TWO ranges (lo1=[0..x], lo2=[y..179]), matching the project's
       two-range red convention.
    3. Optional separation — for any pair of labels marked as "should separate",
       the per-channel boundaries are pushed to the midpoint of any overlap,
       maximising their distinctness.
"""

import cv2
import numpy as np
import json
import sys
import os

# ── Tuning ────────────────────────────────────────────────────────────────────
WINDOW_W        = 1500
SIDEBAR_W       = 380
BASE_RADIUS     = 3.5    # world-pixel radius at zoom=1.0; scales as BASE_RADIUS/zoom
HUE_WRAP_THRESH = 90     # if H samples span > this, treat as wrap-around (e.g. red)
DEFAULT_TRIM    = 10     # percentile trimmed from each end (keep middle 80%)

PRESETS = {
    "ball_align": ["ball", "pedestal", "background"],
    "sniper":     ["red_ball", "blue_ball", "background"],
    "custom":     [],
}

LABEL_COLORS_BGR = [
    (0,   220,   0),   # green
    (0,   100, 255),   # orange
    (255,  60,  60),   # blue
    (0,   200, 255),   # yellow
    (180,  50, 255),   # purple
    (0,   255, 200),   # lime
    (80,   80, 255),   # red
    (255, 130,   0),   # cyan
    (120,   0, 200),   # maroon
    (200, 200,   0),   # teal
]

# ── State ─────────────────────────────────────────────────────────────────────
images       = []    # list of BGR ndarray
image_names  = []
img_idx      = 0

labels       = []    # list of str
pins         = []    # [{"img":int, "x":int, "y":int, "li":int}]
active_label = 0
sep_pairs    = set() # frozensets of two label indices that should be separated

trim_pct     = DEFAULT_TRIM   # percentile trimmed each end
results      = {}             # last computed results dict
needs_redraw = True

pan_x  = 0
pan_y  = 0
zoom   = 1.0
_drag  = None   # (start_sx, start_sy, start_pan_x, start_pan_y)

# ── Coordinate helpers ────────────────────────────────────────────────────────
def w2s(x, y):
    return int(x * zoom + pan_x), int(y * zoom + pan_y)

def s2w(sx, sy):
    return int((sx - pan_x) / zoom), int((sy - pan_y) / zoom)

# ── Zoom-adaptive sampling ────────────────────────────────────────────────────
def _sample_radius():
    """World-pixel radius to average around each pin, inversely proportional to zoom."""
    return max(1, int(BASE_RADIUS / zoom))

def sample_hsv_at(img_bgr, wx, wy):
    """
    Return an Nx3 float32 array of HSV samples from a zoom-adaptive neighbourhood
    around world pixel (wx, wy).  Empty array if out of bounds.
    """
    r = _sample_radius()
    h, w = img_bgr.shape[:2]
    x0, x1 = max(0, wx-r), min(w, wx+r+1)
    y0, y1 = max(0, wy-r), min(h, wy+r+1)
    patch = img_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    return hsv.reshape(-1, 3)

# ── Hue wrap handling ─────────────────────────────────────────────────────────
def _unwrap_hue(h_vals):
    """
    If hue samples span > HUE_WRAP_THRESH (e.g. red straddling 0/180),
    shift values so the distribution is contiguous.
    Returns (shifted_h_vals, shift_amount) where shift_amount is 0 or 90.
    Caller must apply modulo 180 when converting back.
    """
    if len(h_vals) == 0:
        return h_vals, 0
    span = np.max(h_vals) - np.min(h_vals)
    if span <= HUE_WRAP_THRESH:
        return h_vals, 0
    # Shift values below the midpoint up by 180 so red (175–5) becomes (175–185)
    shifted = h_vals.copy()
    mid = (np.max(h_vals) + np.min(h_vals)) / 2.0
    shifted[shifted < mid] += 180.0
    return shifted, 180

def _hue_ranges_from_samples(h_vals, p_lo, p_hi):
    """
    Given raw hue samples, return either one or two [lo, hi] pairs.
    Two pairs are returned when the hue wraps around 0/180 (e.g. red).
    Each lo/hi is an integer in [0, 179].
    """
    if len(h_vals) == 0:
        return [(0, 0)]
    h_shifted, shift = _unwrap_hue(h_vals)
    lo_s = np.percentile(h_shifted, p_lo)
    hi_s = np.percentile(h_shifted, p_hi)
    if shift == 0:
        return [(int(np.clip(lo_s, 0, 179)), int(np.clip(hi_s, 0, 179)))]
    # Wrap: split into two ranges
    # lo_s..hi_s in shifted space; any value >= 180 wraps back to 0..
    lo_clipped = lo_s % 180
    hi_clipped = hi_s % 180
    if hi_s < 180:
        # Both endpoints are in the normal range — single range after all
        return [(int(np.clip(lo_clipped, 0, 179)), int(np.clip(hi_clipped, 0, 179)))]
    # Genuine wrap: [lo_clipped..179] and [0..hi_clipped]
    return [
        (int(np.clip(lo_clipped, 0, 179)), 179),
        (0, int(np.clip(hi_clipped, 0, 179))),
    ]

# ── Core range computation ────────────────────────────────────────────────────
def _collect_samples():
    """Return dict {label_idx: Nx3 float32 HSV array}."""
    by_label = {i: [] for i in range(len(labels))}
    for pin in pins:
        arr = sample_hsv_at(images[pin["img"]], pin["x"], pin["y"])
        if len(arr):
            by_label[pin["li"]].append(arr)
    return {li: np.concatenate(v) if v else np.empty((0, 3), dtype=np.float32)
            for li, v in by_label.items()}

def _percentile_range(vals, p_lo, p_hi):
    """1-D array → (lo, hi) after trimming."""
    if len(vals) == 0:
        return 0.0, 0.0
    return float(np.percentile(vals, p_lo)), float(np.percentile(vals, p_hi))

def _separate_1d(lo_a, hi_a, lo_b, hi_b):
    """
    Push two 1-D ranges apart at their overlap midpoint.
    Returns (new_lo_a, new_hi_a, new_lo_b, new_hi_b).
    """
    if hi_a < lo_b or hi_b < lo_a:
        return lo_a, hi_a, lo_b, hi_b   # no overlap
    if lo_a <= lo_b:
        mid = (hi_a + lo_b) / 2.0
        return lo_a, mid, mid, hi_b
    else:
        mid = (hi_b + lo_a) / 2.0
        return mid, hi_a, lo_b, mid

def compute_ranges():
    """
    Main computation.  Returns dict keyed by label name:
        {
          "n_samples": int,
          "h_ranges": [(lo, hi)] or [(lo1,hi1),(lo2,hi2)] for wrap-around,
          "s_lo": int, "s_hi": int,
          "v_lo": int, "v_hi": int,
          "wrapped": bool,
        }
    Separation pairs get their S and V (and single H) pushed apart at midpoints.
    """
    p_lo = trim_pct
    p_hi = 100 - trim_pct
    samples = _collect_samples()

    # Per-label base ranges
    base = {}
    for li, arr in samples.items():
        if li >= len(labels) or len(arr) == 0:
            continue
        h_vals = arr[:, 0]
        s_lo, s_hi = _percentile_range(arr[:, 1], p_lo, p_hi)
        v_lo, v_hi = _percentile_range(arr[:, 2], p_lo, p_hi)
        h_ranges = _hue_ranges_from_samples(h_vals, p_lo, p_hi)
        base[li] = {
            "h_ranges": h_ranges,
            "s_lo": s_lo, "s_hi": s_hi,
            "v_lo": v_lo, "v_hi": v_hi,
            "wrapped": len(h_ranges) == 2,
            "n": len(arr),
        }

    # Apply separation: push S and V boundaries apart for each sep pair
    # (H is harder to push when wrapped, so we focus on S and V which are
    # the most reliable separators for things like white ball vs blue pedestal)
    for pair in sep_pairs:
        pair_list = sorted(pair)
        if len(pair_list) != 2:
            continue
        a, b = pair_list
        if a not in base or b not in base:
            continue
        # Separate S
        base[a]["s_lo"], base[a]["s_hi"], base[b]["s_lo"], base[b]["s_hi"] = \
            _separate_1d(base[a]["s_lo"], base[a]["s_hi"],
                         base[b]["s_lo"], base[b]["s_hi"])
        # Separate V
        base[a]["v_lo"], base[a]["v_hi"], base[b]["v_lo"], base[b]["v_hi"] = \
            _separate_1d(base[a]["v_lo"], base[a]["v_hi"],
                         base[b]["v_lo"], base[b]["v_hi"])
        # Separate H only if both are single-range (non-wrapped)
        if not base[a]["wrapped"] and not base[b]["wrapped"]:
            h_a = base[a]["h_ranges"][0]
            h_b = base[b]["h_ranges"][0]
            new_a_lo, new_a_hi, new_b_lo, new_b_hi = \
                _separate_1d(h_a[0], h_a[1], h_b[0], h_b[1])
            base[a]["h_ranges"] = [(new_a_lo, new_a_hi)]
            base[b]["h_ranges"] = [(new_b_lo, new_b_hi)]

    # Clip everything to valid ranges
    def ci(v, lo=0, hi=255): return int(np.clip(round(v), lo, hi))

    out = {}
    for li, rng in base.items():
        name = labels[li]
        h_ranges_int = [(ci(lo, 0, 179), ci(hi, 0, 179))
                        for lo, hi in rng["h_ranges"]]
        out[name] = {
            "n_samples": rng["n"],
            "h_ranges":  h_ranges_int,
            "s_lo":  ci(rng["s_lo"]),  "s_hi": ci(rng["s_hi"]),
            "v_lo":  ci(rng["v_lo"]),  "v_hi": ci(rng["v_hi"]),
            "wrapped": rng["wrapped"],
        }
    return out

def format_config_lines(res):
    """Return list of copy-paste config.py strings."""
    lines = []
    for name, rng in res.items():
        safe = name.upper().replace(" ", "_").replace("-", "_")
        n    = rng["n_samples"]
        s_lo, s_hi = rng["s_lo"], rng["s_hi"]
        v_lo, v_hi = rng["v_lo"], rng["v_hi"]
        lines.append(f"# {name}  ({n} samples, trim={trim_pct}%)")
        if rng["wrapped"]:
            (h1l, h1h), (h2l, h2h) = rng["h_ranges"][0], rng["h_ranges"][1]
            lines.append(f"CAM_4K_BALL_{safe} = [")
            lines.append(f"    (np.array([{h1l}, {s_lo}, {v_lo}]),  np.array([{h1h}, {s_hi}, {v_hi}])),")
            lines.append(f"    (np.array([{h2l}, {s_lo}, {v_lo}]),  np.array([{h2h}, {s_hi}, {v_hi}])),")
            lines.append(f"]")
        else:
            hl, hh = rng["h_ranges"][0]
            lines.append(f"CAM_4K_BALL_{safe}_LO = np.array([{hl}, {s_lo}, {v_lo}])")
            lines.append(f"CAM_4K_BALL_{safe}_HI = np.array([{hh}, {s_hi}, {v_hi}])")
        lines.append("")
    return lines

def save_results(res):
    path = "hsv_results.json"
    # Make JSON-serialisable
    serial = {}
    for name, rng in res.items():
        serial[name] = {k: (list(v) if isinstance(v, np.ndarray) else v)
                        for k, v in rng.items()}
    with open(path, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"\n[hsv_calibrator] Saved → {path}")
    print("\n── config.py lines ─────────────────────────────────────────────")
    for ln in format_config_lines(res):
        print(ln)

# ── Drawing ───────────────────────────────────────────────────────────────────
def _draw():
    img_area_w = WINDOW_W - SIDEBAR_W
    canvas_h   = 760
    canvas     = np.full((canvas_h, WINDOW_W, 3), (20, 20, 28), dtype=np.uint8)

    # Image
    if images:
        frame = images[img_idx]
        fh, fw = frame.shape[:2]
        sw, sh = int(fw * zoom), int(fh * zoom)
        if sw > 0 and sh > 0:
            scaled = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
            sx0, sy0 = max(0, pan_x), max(0, pan_y)
            ox,  oy  = max(0, -pan_x), max(0, -pan_y)
            cw = min(sw - ox, img_area_w - sx0)
            ch = min(sh - oy, canvas_h   - sy0)
            if cw > 0 and ch > 0:
                canvas[sy0:sy0+ch, sx0:sx0+cw] = scaled[oy:oy+ch, ox:ox+cw]

    # Sampling radius ring (shows how big the sample area is)
    r_world = _sample_radius()
    r_screen = max(1, int(r_world * zoom))
    cv2.putText(canvas, f"sample r={r_world}px", (8, canvas_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 110), 1, cv2.LINE_AA)

    # Pins
    for pin in pins:
        if pin["img"] != img_idx:
            continue
        sx, sy = w2s(pin["x"], pin["y"])
        if 0 <= sx < img_area_w and 0 <= sy < canvas_h:
            col = LABEL_COLORS_BGR[pin["li"] % len(LABEL_COLORS_BGR)]
            # Draw sample radius ring
            cv2.circle(canvas, (sx, sy), r_screen, col, 1)
            # Filled centre dot
            cv2.circle(canvas, (sx, sy), 5, col, -1)
            cv2.circle(canvas, (sx, sy), 5, (255, 255, 255), 1)
            lbl = labels[pin["li"]] if pin["li"] < len(labels) else "?"
            cv2.putText(canvas, lbl[0].upper(), (sx + 7, sy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    # Divider
    cv2.line(canvas, (img_area_w, 0), (img_area_w, canvas_h), (45, 45, 65), 1)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    x0 = img_area_w + 10
    y  = 18
    W  = SIDEBAR_W - 18

    def txt(s, col=(200, 200, 225), sz=0.50, bold=False):
        nonlocal y
        cv2.putText(canvas, s, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                    sz, col, 2 if bold else 1, cv2.LINE_AA)
        y += int(sz * 38)

    def gap(n=6): nonlocal y; y += n
    def rule():
        nonlocal y
        cv2.line(canvas, (x0, y), (x0+W, y), (50, 50, 70), 1)
        y += 7

    txt("HSV CALIBRATOR", (80, 220, 255), 0.62, bold=True)
    gap(2)
    if images:
        n_here = sum(1 for p in pins if p["img"] == img_idx)
        txt(f"Image {img_idx+1}/{len(images)}: "
            f"{os.path.basename(image_names[img_idx])}",
            (150, 150, 170), 0.43)
        txt(f"zoom {zoom:.1f}×  pan({pan_x},{pan_y})  "
            f"pins {n_here}/{len(pins)}", (110, 110, 140), 0.40)
    gap(6); rule()

    # Labels
    txt("LABELS", (180, 200, 230), 0.48, bold=True)
    label_y_start = y
    for li, lbl in enumerate(labels):
        col = LABEL_COLORS_BGR[li % len(LABEL_COLORS_BGR)]
        n   = sum(1 for p in pins if p["li"] == li)
        active = li == active_label
        bk = (38, 55, 38) if active else (25, 25, 35)
        cv2.rectangle(canvas, (x0-3, y-14), (x0+W, y+10), bk, -1)
        prefix = "▶" if active else " "
        txt(f"{prefix}[{li+1}] {lbl}  ·{n}px", col, 0.48)
    gap(4)
    txt("A=add  D=del active", (110, 110, 140), 0.40)
    gap(6); rule()

    # Trim percentile
    txt(f"TRIM: ±{trim_pct}%  (keep middle {100-2*trim_pct}%)",
        (180, 200, 230), 0.48, bold=True)
    txt("← → to adjust", (110, 110, 140), 0.40)
    gap(6); rule()

    # Separation pairs
    txt("SEPARATION PAIRS  (X to toggle)", (180, 200, 230), 0.48, bold=True)
    txt("Active label ↔ another, must separate:", (130, 130, 155), 0.42)
    for li, lbl in enumerate(labels):
        if li == active_label:
            continue
        pair = frozenset({active_label, li})
        on   = pair in sep_pairs
        col  = (80, 255, 140) if on else (110, 110, 140)
        tick = "■" if on else "□"
        txt(f"  {tick} {labels[active_label]} ↔ {lbl}", col, 0.44)
    gap(2)
    txt("(X cycles through labels to toggle)", (90, 90, 120), 0.38)
    gap(6); rule()

    # Controls
    txt("CONTROLS", (180, 200, 230), 0.46, bold=True)
    for line in [
        "L-click  place pin",
        "R-click  remove pin",
        "Scroll   zoom in/out",
        "Mid-drag pan",
        "N        load/cycle images",
        "ENTER    compute",
        "S        save results",
        "Q/Esc    quit",
    ]:
        txt(line, (130, 130, 155), 0.41)
    gap(6); rule()

    # Results preview
    if results:
        txt("RESULTS", (80, 255, 160), 0.50, bold=True)
        for name, rng in results.items():
            col_r = (160, 230, 160)
            wrap_tag = " (2-range)" if rng.get("wrapped") else ""
            txt(f"{name}{wrap_tag}  n={rng['n_samples']}", col_r, 0.46, bold=True)
            for hlo, hhi in rng["h_ranges"]:
                txt(f"  H [{hlo}–{hhi}]  "
                    f"S [{rng['s_lo']}–{rng['s_hi']}]  "
                    f"V [{rng['v_lo']}–{rng['v_hi']}]",
                    (140, 210, 140), 0.41)
            gap(2)

    return canvas

# ── Mouse ─────────────────────────────────────────────────────────────────────
def on_mouse(event, sx, sy, flags, _):
    global active_label, needs_redraw, pan_x, pan_y, zoom, _drag
    img_area_w = WINDOW_W - SIDEBAR_W

    # Sidebar label-row click
    if event == cv2.EVENT_LBUTTONDOWN and sx >= img_area_w:
        # Rough heuristic: labels start at y=94 (after header rows), 19px each
        rel = sy - 94
        if rel >= 0:
            li = rel // 19
            if 0 <= li < len(labels):
                active_label = li
                needs_redraw = True
        return

    if sx >= img_area_w:
        return

    wx, wy = s2w(sx, sy)

    if event == cv2.EVENT_LBUTTONDOWN:
        if not images: return
        pins.append({"img": img_idx, "x": wx, "y": wy, "li": active_label})
        needs_redraw = True

    elif event == cv2.EVENT_RBUTTONDOWN:
        best_d, best_i = 9999, None
        for i, p in enumerate(pins):
            if p["img"] != img_idx: continue
            psx, psy = w2s(p["x"], p["y"])
            d = ((psx-sx)**2 + (psy-sy)**2) ** 0.5
            if d < best_d: best_d, best_i = d, i
        if best_i is not None and best_d < 16:
            pins.pop(best_i)
            needs_redraw = True

    elif event == cv2.EVENT_MBUTTONDOWN:
        _drag = (sx, sy, pan_x, pan_y)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_MBUTTON) and _drag:
        dx, dy = sx - _drag[0], sy - _drag[1]
        pan_x, pan_y = _drag[2] + dx, _drag[3] + dy
        needs_redraw = True

    elif event == cv2.EVENT_MBUTTONUP:
        _drag = None

    elif event == cv2.EVENT_MOUSEWHEEL:
        old_z = zoom
        zoom = min(zoom * 1.13, 12.0) if flags > 0 else max(zoom / 1.13, 0.08)
        pan_x = int(sx - (sx - pan_x) * (zoom / old_z))
        pan_y = int(sy - (sy - pan_y) * (zoom / old_z))
        needs_redraw = True

# ── Image loading ─────────────────────────────────────────────────────────────
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[!] Cannot load: {path}")
        return False
    images.append(img); image_names.append(path)
    print(f"Loaded: {path}  {img.shape[1]}×{img.shape[0]}")
    return True

def load_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        paths = filedialog.askopenfilenames(
            title="Open image(s)",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp"), ("All","*")])
        root.destroy()
        return list(paths)
    except Exception:
        p = input("Image path: ").strip()
        return [p] if p else []

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global img_idx, active_label, labels, needs_redraw, trim_pct, results
    global pan_x, pan_y, zoom

    print("═"*55)
    print("  HSV Calibrator")
    print("═"*55)

    # Preset selection
    pk = list(PRESETS.keys())
    print("\nPresets:")
    for i, k in enumerate(pk, 1):
        print(f"  {i}. {k}  {PRESETS[k]}")
    print("  0. Empty")
    c = input("Choose [0]: ").strip() or "0"
    try:
        ci = int(c)
        labels = list(PRESETS[pk[ci-1]]) if 1 <= ci <= len(pk) else []
    except (ValueError, IndexError):
        labels = []

    # Images
    for p in sys.argv[1:]:
        load_image(p)
    if not images:
        print("No images provided — opening dialog…")
        for p in load_dialog():
            load_image(p)
    if not images:
        print("No images loaded. Exiting."); return

    cv2.namedWindow("HSV Calibrator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Calibrator", WINDOW_W, 760)
    cv2.setMouseCallback("HSV Calibrator", on_mouse)

    # X-key cycle state: which label to toggle separation against active
    _x_cycle = [0]

    while True:
        if needs_redraw:
            cv2.imshow("HSV Calibrator", _draw())
            needs_redraw = False

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break

        elif key == 13:   # Enter — compute
            if not pins:
                print("[!] No pins placed yet.")
            else:
                results = compute_ranges()
                print("\n── Computed HSV ranges ─────────────────────────────")
                for name, rng in results.items():
                    wrap = " (wrap→2 ranges)" if rng["wrapped"] else ""
                    print(f"  {name}{wrap}  n={rng['n_samples']}  trim={trim_pct}%")
                    for hlo, hhi in rng["h_ranges"]:
                        print(f"    H[{hlo}–{hhi}]  "
                              f"S[{rng['s_lo']}–{rng['s_hi']}]  "
                              f"V[{rng['v_lo']}–{rng['v_hi']}]")
                print()
                needs_redraw = True

        elif key == ord('s'):
            if results: save_results(results)
            else: print("[!] Press ENTER first.")

        elif key == ord('n'):
            paths = load_dialog()
            for p in paths: load_image(p)
            if len(images) > 1:
                img_idx = (img_idx + 1) % len(images)
            pan_x = pan_y = 0; zoom = 1.0
            needs_redraw = True

        elif key == ord('a'):
            name = input("New label name: ").strip()
            if name and name not in labels:
                labels.append(name)
                active_label = len(labels) - 1
                needs_redraw = True

        elif key == ord('d'):
            if labels:
                rm = active_label
                labels.pop(rm)
                # Remove pins for deleted label, shift indices above it
                pins[:] = [p for p in pins if p["li"] != rm]
                for p in pins:
                    if p["li"] > rm: p["li"] -= 1
                # Fix sep_pairs
                new_pairs = set()
                for pair in sep_pairs:
                    pl = sorted(pair)
                    if rm in pl: continue
                    new_pairs.add(frozenset(
                        li if li < rm else li-1 for li in pl))
                sep_pairs.clear(); sep_pairs.update(new_pairs)
                active_label = min(active_label, max(0, len(labels)-1))
                needs_redraw = True

        elif key == ord('x'):
            # Cycle through other labels and toggle separation pair with active
            others = [li for li in range(len(labels)) if li != active_label]
            if others:
                _x_cycle[0] = (_x_cycle[0] + 1) % len(others)
                other = others[_x_cycle[0]]
                pair  = frozenset({active_label, other})
                if pair in sep_pairs:
                    sep_pairs.discard(pair)
                    print(f"Separation OFF: {labels[active_label]} ↔ {labels[other]}")
                else:
                    sep_pairs.add(pair)
                    print(f"Separation ON:  {labels[active_label]} ↔ {labels[other]}")
                needs_redraw = True

        elif key in (81, 2):   # left arrow
            trim_pct = max(0, trim_pct - 1)
            print(f"trim={trim_pct}%  (keep middle {100-2*trim_pct}%)")
            needs_redraw = True

        elif key in (83, 3):   # right arrow
            trim_pct = min(49, trim_pct + 1)
            print(f"trim={trim_pct}%  (keep middle {100-2*trim_pct}%)")
            needs_redraw = True

        elif ord('1') <= key <= ord('9'):
            li = key - ord('1')
            if li < len(labels):
                active_label = li; needs_redraw = True

        if cv2.getWindowProperty("HSV Calibrator", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("[hsv_calibrator] Done.")

if __name__ == "__main__":
    main()
