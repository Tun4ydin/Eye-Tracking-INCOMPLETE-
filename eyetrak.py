import cv2
import dlib
import numpy as np
import mss
import json
import os
import time
import datetime

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

sct = mss.mss()
monitor = sct.monitors[1]
sw, sh = monitor["width"], monitor["height"]

cap = cv2.VideoCapture(0)
json_buffer = []

def get_face_data(frame_gray, face):
    shape = predictor(frame_gray, face)
    eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 48)])
    head_center = (shape.part(27).x, shape.part(27).y) # Nose bridge
    return np.mean(eye_pts[:, 0]), np.mean(eye_pts[:, 1]), head_center

cv2.namedWindow("Main", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Main", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for i in range(2, 0, -1):
    bg = np.zeros((sh, sw, 3), dtype=np.uint8)
    cv2.putText(bg, f"Starting Calibration in {i}...", (sw//2-200, sh//2), 1, 3, (255, 255, 255), 3)
    cv2.imshow("Main", bg)
    cv2.waitKey(1000)

corners = [("TL", (100, 100)), ("TR", (sw-100, 100)), ("BR", (sw-100, sh-100)), 
           ("BL", (100, sh-100)), ("CENTER", (sw//2, sh//2))]

eye_samples = {"x": [], "y": []}
head_samples = []

for name, pos in corners:
    start_time = time.time()
    while time.time() - start_time < 2:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        bg = np.zeros((sh, sw, 3), dtype=np.uint8)
        cv2.circle(bg, pos, 30, (0, 0, 255), -1)
        cv2.circle(bg, pos, 10, (255, 255, 255), -1)
        cv2.imshow("Main", bg)
        if faces:
            ex, ey, hc = get_face_data(gray, faces[0])
            eye_samples["x"].append(ex); eye_samples["y"].append(ey)
            head_samples.append(hc)
        cv2.waitKey(1)

c_left, c_right = min(eye_samples["x"]), max(eye_samples["x"])
c_top, c_bottom = min(eye_samples["y"]), max(eye_samples["y"])
calib_head_pos = np.mean(head_samples, axis=0)

def get_rating(samples, targets, sw, sh, cl, cr, ct, cb):
    errs = []
    chunk = len(samples["x"]) // 5
    for i, (_, t_pos) in enumerate(targets):
        mx = 1 - ((np.mean(samples["x"][i*chunk:(i+1)*chunk]) - cl) / (cr - cl + 1e-6))
        my = (np.mean(samples["y"][i*chunk:(i+1)*chunk]) - ct) / (cb - ct + 1e-6)
        dist = np.sqrt((mx*sw - t_pos[0])**2 + (my*sh - t_pos[1])**2)
        errs.append(dist)
    avg_e = np.mean(errs)
    score = 5 if avg_e < 85 else 4 if avg_e < 180 else 3 if avg_e < 300 else 2 if avg_e < 500 else 1
    return score, avg_e

score, error_px = get_rating(eye_samples, corners, sw, sh, c_left, c_right, c_top, c_bottom)

bg = np.zeros((sh, sw, 3), dtype=np.uint8)
cv2.putText(bg, f"CALIBRATION COMPLETE", (sw//2-200, sh//2-100), 1, 3, (0, 255, 0), 3)
cv2.putText(bg, f"Rating: {score}/5", (sw//2-100, sh//2), 1, 2, (255, 255, 255), 2)
cv2.putText(bg, f"Offset: {error_px:.1f}px", (sw//2-100, sh//2+50), 1, 2, (255, 255, 255), 2)
cv2.imshow("Main", bg)
cv2.waitKey(3000)
cv2.destroyWindow("Main")

out = cv2.VideoWriter('eye_tracker_final.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (sw, sh))

while True:
    screen_frame = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
    ret, web_frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(web_frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if faces:
        ex, ey, current_hc = get_face_data(gray, faces[0])
        head_drift = float(np.linalg.norm(np.array(current_hc) - calib_head_pos))
        is_fixed = head_drift < 45 # Threshold for excessive movement
        
        px = 1 - ((ex - c_left) / (c_right - c_left + 1e-6))
        py = (ey - c_top) / (c_bottom - c_top + 1e-6)
        gx, gy = max(0, min(sw, px*sw)), max(0, min(sh, py*sh))

        entry = {
            "category": "tracker", "request": "get", "statuscode": 200,
            "values": {
                "frame": {
                    "avg": {"x": gx, "y": gy},
                    "fix": is_fixed,
                    "lefteye": {"avg": {"x": gx-10, "y": gy}, "pcenter": {"x": px, "y": py}, "psize": 30, "raw": {"x": ex, "y": ey}},
                    "raw": {"x": gx, "y": gy},
                    "state": 7 if is_fixed else 0,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "calibration_quality": score,
                    "head_movement_drift": round(head_drift, 2),
                    "excessive_movement": not is_fixed
                }
            }
        }
        json_buffer.append(entry)
        
        color = (0, 255, 0) if is_fixed else (0, 0, 255)
        cv2.circle(screen_frame, (int(gx), int(gy)), 15, color, -1)
        if not is_fixed:
            cv2.putText(screen_frame, "MOVE HEAD BACK", (sw//2-150, 100), 1, 2, (0,0,255), 3)

    out.write(screen_frame)
    cv2.imshow('Tracking (Q to Stop)', cv2.resize(screen_frame, (sw//4, sh//4)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

with open('eye_data.json', 'w') as f:
    json.dump(json_buffer, f, indent=4)

cap.release(); out.release(); cv2.destroyAllWindows()