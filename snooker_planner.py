import numpy as np
import json
import math
import matplotlib.pyplot as plt

# Import the vision function directly from your main script
from main import detect_balls_in_zones

# ==========================================
# TABLE & ROBOT ENVIRONMENT SETUP (In Millimeters)
# ==========================================
TABLE_WIDTH_X = 1200   # Short axis (width) as specified
TABLE_LENGTH_Y = 3600  # Total length (2400mm green + 1200mm white)
BALL_DIAMETER = 52.5   # Full diameter of Aramith balls (52.5 mm)

ROBOT_SIZE = 450
ROBOT_HALF = ROBOT_SIZE / 2.0

# Robot Walkable Width boundaries (X Axis)
ROBOT_X_MIN = ROBOT_HALF + 20       
ROBOT_X_MAX = TABLE_WIDTH_X - ROBOT_HALF - 20 

# Robot Walkable Length boundaries (Y Axis)
# "starts at y 210, length 150 to the back" -> Y from 2100 to 3600
ROBOT_SHOOT_Y = 2100 + ROBOT_HALF + 50   # Centerline axis line for robot shooting position

# 5 Hole Target Locations (0,0 is at Hole 3 / top-left corner)
POCKETS = {
    "Hole_3": (0, 0),         # Top-Left Corner (closest to C)
    "Hole_4": (600, 0),       # Top-Middle
    "Hole_5": (1200, 0),      # Top-Right Corner
    "Hole_2": (0, 1050),      # Left-Side Pocket near foul area
    "Hole_1": (1200, 1050)    # Right-Side Pocket near foul area
}

# ─────────────────────────────────────────────────────────────────────────────
# CORNER POCKET OFFSET
#
# When shooting at a corner pocket (Hole_3 or Hole_5) from a shallow angle,
# aiming at the pocket CENTER risks clipping the side wall.
# This offset shifts the effective aim point inward along the pocket edge:
#   Hole_3 (left corner, x=0)    → effective aim at (CORNER_OFFSET_MM, 0)
#   Hole_5 (right corner, x=1200)→ effective aim at (1200 - CORNER_OFFSET_MM, 0)
#
# The planner computes ALL geometry (ghost ball, robot rx, firing angle)
# using this effective aim point. The sniper_4k.py angle system works correctly
# with this because it uses the real pocket position to compute target_relative_angle,
# while fire_angle_deg encodes the correct direction toward the offset point.
# The sniper's STEP 2 (lateral pixel check) also uses this same offset.
#
# Set to 0 to disable. Must be kept in sync with CORNER_OFFSET_MM in sniper_4k.py.
# ─────────────────────────────────────────────────────────────────────────────
CORNER_OFFSET_MM = 25.0   # mm inward from corner pocket center; 0 to ~50mm typical

CORNER_EFFECTIVE_POCKETS = {
    "Hole_3": (CORNER_OFFSET_MM, 0),
    "Hole_5": (1200 - CORNER_OFFSET_MM, 0),
}

# Swapped and re-mapped coordinates for Placement Zones A-G (X=width, Y=length)
REAL_ZONES = {
    "A": (1050, 1050),
    "B": (150, 1050), # "B": (120, 1100),
    "C": (200, 300),  # Closest zone to Hole 3 (0,0)
    "D": (400, 300),
    "E": (600, 500),
    "F": (800, 300),
    "G": (1000, 300)
}

def calculate_ghost_ball(target_pos, pocket_pos):
    """Calculates ghost ball exactly one ball diameter behind target on the pocket line."""
    tx, ty = target_pos
    px, py = pocket_pos
    
    dx = tx - px
    dy = ty - py
    distance = math.hypot(dx, dy)
    
    if distance == 0:
        return target_pos
        
    ux = dx / distance
    uy = dy / distance
    
    gx = tx + (ux * BALL_DIAMETER)
    gy = ty + (uy * BALL_DIAMETER)
    return (round(gx, 2), round(gy, 2))

def find_best_shot(zone_name, target_pos):
    tx, ty = target_pos
    best_pocket = None
    best_ghost = None
    best_robot_x = None
    best_score = float('inf')
    best_cut_angle = 0.0
    
    # HYBRID SCORING WEIGHTS
    # Distance is in mm (e.g. 500mm to 2000mm)
    # Angle is in degrees (e.g. 0 to 90 degrees)
    WEIGHT_DISTANCE = 1.0
    WEIGHT_ANGLE = 25.0  # 1 degree of cut angle adds 30 "units" of difficulty (equivalent to 30mm extra distance)
    MAX_CUT_ANGLE = 55.0 # If the cut angle is greater than this, heavily penalize it
    
    for pocket_name, pocket_pos in POCKETS.items():
        # Use effective (offset) pocket for corner shots; real pocket otherwise.
        # All geometry (ghost ball, robot position, firing angle) is computed
        # from the effective aim point. The real pocket is kept in target_pocket
        # for the sniper to look up the correct AprilTag.
        eff_pocket_pos = CORNER_EFFECTIVE_POCKETS.get(pocket_name, pocket_pos)

        px, py = pocket_pos
        epx, epy = eff_pocket_pos
        gx, gy = calculate_ghost_ball(target_pos, eff_pocket_pos)
        
        # Shot vector (uses effective pocket direction)
        vx_shot = epx - tx
        vy_shot = epy - ty
        if vx_shot == 0 and vy_shot == 0: continue
        
        # Find the mathematical ideal Robot X for a perfect 0-degree straight shot
        if vy_shot != 0:
            t = (ROBOT_SHOOT_Y - gy) / vy_shot
            ideal_rx = gx + t * vx_shot
        else:
            ideal_rx = gx
            
        # Clamp robot X to the physical walkable constraints of the robot
        clamped_rx = max(ROBOT_X_MIN, min(ROBOT_X_MAX, ideal_rx))
        
        # Aim vector (Robot to Ghost)
        vx_aim = gx - clamped_rx
        vy_aim = gy - ROBOT_SHOOT_Y
        
        # Calculate precise Cut Angle using vector dot product
        mag_aim = math.hypot(vx_aim, vy_aim)
        mag_shot = math.hypot(vx_shot, vy_shot)
        
        if mag_aim == 0 or mag_shot == 0: continue
        
        dot_product = (vx_aim * vx_shot) + (vy_aim * vy_shot)
        # Clamp dot_product to avoid floating point math domain errors in acos
        cos_angle = max(-1.0, min(1.0, dot_product / (mag_aim * mag_shot)))
        cut_angle = math.degrees(math.acos(cos_angle))
        
        # Distance: use real pocket for scoring (actual difficulty of the shot)
        dist = math.hypot(tx - px, ty - py)
        
        # HYBRID COST FUNCTION SCORE
        shot_difficulty_score = (dist * WEIGHT_DISTANCE) + (cut_angle * WEIGHT_ANGLE)
        
        # Impossible / extremely risky shot penalty
        if cut_angle > MAX_CUT_ANGLE:
            shot_difficulty_score += 10000.0 # Massive penalty, but keeps it as a last-resort fallback
        
        if shot_difficulty_score < best_score:
            best_score = shot_difficulty_score
            best_pocket = pocket_name
            best_ghost = (gx, gy)
            best_robot_x = round(clamped_rx, 2)
            best_cut_angle = round(cut_angle, 1)
                
    return best_pocket, best_ghost, (best_robot_x, ROBOT_SHOOT_Y), best_cut_angle

def visualize_table_plan(shooting_plan):
    fig, ax = plt.subplots(figsize=(6, 10))
    
    ax.set_xlim(-100, 1300)
    ax.set_ylim(3700, -100)
    
    rect_green = plt.Rectangle((0, 0), 1200, 2400, linewidth=2, edgecolor='#5d4037', facecolor='#2e7d32', alpha=0.85)
    ax.add_patch(rect_green)
    
    rect_white = plt.Rectangle((0, 2400), 1200, 1200, linewidth=2, edgecolor='black', facecolor='#f5f5f5', alpha=0.9)
    ax.add_patch(rect_white)
    
    ax.axhline(y=2100, color='red', linestyle='--', linewidth=2, label='Foul Line Bound')
    ax.axhline(y=ROBOT_SHOOT_Y, color='blue', linestyle=':', linewidth=1.5, label='Robot Travel Line')
    
    for name, (px, py) in POCKETS.items():
        ax.plot(px, py, 'ko', markersize=16)
        ax.text(px, py - 45, name, fontsize=8, ha='center', fontweight='bold')
        
    for zone, (zx, zy) in REAL_ZONES.items():
        ax.plot(zx, zy, 'yo', markersize=6, markeredgecolor='black')
        ax.text(zx, zy + 50, zone, fontsize=8, ha='center', color='yellow', fontweight='bold')
        
    for zone, info in shooting_plan.items():
        tx, ty = info["target_ball_real_xy"]
        gx, gy = info["ghost_ball_xy"]
        rx, ry = info["robot_alignment_xy"]
        ball_color = 'red' if info["color"] == 'Red' else 'blue'
        
        ax.plot(tx, ty, 'o', color=ball_color, markersize=11, markeredgecolor='black', label=f'Target ({info["color"]})')
        ax.plot(gx, gy, 'o', color='white', markersize=11, markeredgecolor='black', linestyle='None')
        ax.plot(gx, gy, 'x', color='black', markersize=6)
        ax.plot(rx, ry, 's', color='purple', markersize=12, label='Robot Center')
        
        ax.annotate('', xy=(gx, gy), xytext=(rx, ry), arrowprops=dict(arrowstyle="->", color='purple', lw=1.5))
        px, py = POCKETS[info["target_pocket"]]
        ax.plot([gx, px], [gy, py], color='gold', linestyle='-', linewidth=2, alpha=0.8)
        
    ax.set_aspect('equal')
    plt.title("CARC 2026 Advanced Snooker Robot\nOptimized Low-Precision Shooting Map", fontsize=11, fontweight='bold', pad=10)
    plt.xlabel("Width X (mm)")
    plt.ylabel("Length Y (mm)")
    plt.grid(True, linestyle=':', alpha=0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower left')
    
    plt.tight_layout()
    plt.savefig("robot_shooting_map_2d.png", dpi=300)
    print("[SUCCESS] Top-down 2D strategy view saved as 'robot_shooting_map_2d.png'")
    plt.show()

def generate_shooting_plan(detected_zones):
    shooting_plan = {}
    print("\n" + "="*65)
    print("                ROBOT MISSION PLAN CALCULATOR               ")
    print("="*65)
    
    for zone, status in detected_zones.items():
        if "Ball" in status and "White" not in status:
            color = "Red" if "Red" in status else "Blue"
            target_coords = REAL_ZONES[zone]
            
            pocket, ghost_ball, robot_pos, cut_angle = find_best_shot(zone, target_coords)
            
            # Firing angle = direction from robot toward ghost ball.
            # This is the same as the direction toward the effective pocket,
            # which is already baked into ghost_ball by find_best_shot.
            dx = ghost_ball[0] - robot_pos[0]
            dy = ghost_ball[1] - robot_pos[1]
            angle_deg = round(math.degrees(math.atan2(dy, dx)) % 360, 1)
            
            shooting_plan[zone] = {
                "color": color,
                "target_ball_zone": zone,
                "target_ball_real_xy": target_coords,
                "target_pocket": pocket,
                "ghost_ball_xy": ghost_ball,
                "robot_alignment_xy": robot_pos,
                "firing_angle_deg": angle_deg,
                "cut_angle_deg": cut_angle
            }
            
            print(f"🎯 Target Found in Zone {zone} [{color}]")
            print(f"   ├── Target XY     : {target_coords} mm")
            print(f"   ├── Optimum Pocket: {pocket}")
            if pocket in CORNER_EFFECTIVE_POCKETS:
                eff = CORNER_EFFECTIVE_POCKETS[pocket]
                print(f"   ├── Eff. Aim Point: {eff} mm  (offset {CORNER_OFFSET_MM}mm from corner)")
            print(f"   ├── Drive WheelsTo: {robot_pos} mm")
            print(f"   ├── Aim Actuator  : {angle_deg}° Heading")
            print(f"   └── Cut Angle     : {cut_angle}° (Minimized Deviation)")
            print("-" * 65)

    with open("robot_shooting_plan.json", "w") as f:
        json.dump(shooting_plan, f, indent=4)
    print("[SUCCESS] Configuration map stored in 'robot_shooting_plan.json'")
    
    if shooting_plan:
        visualize_table_plan(shooting_plan)
        
    return shooting_plan

if __name__ == "__main__":
    import sys

    # Allows the HD Camera script to pass the freshly taken photo via command line
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = r"camera_images\img_21_colored.jpg"
    
        # OPTION B: Run using live physical camera acquisition during competition
        # print("Initializing live hardware camera tracking...")
        # live_detections = detect_balls_in_zones() # No arguments triggers live camera frame
    
    print(f"Running vision module with image: {img_path}")
    live_detections = detect_balls_in_zones(img_path)
    if live_detections:
        generate_shooting_plan(live_detections)
    else:
        print("Execution failed: No items returned from vision processor.")