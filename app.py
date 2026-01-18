import cv2
import os
from core.pose_detector import PoseDetector
from core.cloth_warp import warp_cloth
from core.stabilizer import Stabilizer
from utils.alpha_blend import alpha_blend

# Load clothes
CLOTH_DIR = "assets/clothes"
clothes = [os.path.join(CLOTH_DIR, f) for f in os.listdir(CLOTH_DIR)]
cloth_index = 0
cloth = cv2.imread(clothes[cloth_index], cv2.IMREAD_UNCHANGED)

pose_detector = PoseDetector()
stabilizer = Stabilizer()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = pose_detector.get_landmarks(frame)

    if landmarks:
        landmarks = stabilizer.smooth(landmarks)

        dst_pts = [
            landmarks["ls"],
            landmarks["rs"],
            landmarks["lh"],
            landmarks["rh"]
        ]

        warped = warp_cloth(cloth, dst_pts, frame.shape)
        frame = alpha_blend(frame, warped)

    cv2.imshow("AI Virtual Try-On", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        cloth_index = (cloth_index + 1) % len(clothes)
        cloth = cv2.imread(clothes[cloth_index], cv2.IMREAD_UNCHANGED)
    elif key == ord('p'):
        cloth_index = (cloth_index - 1) % len(clothes)
        cloth = cv2.imread(clothes[cloth_index], cv2.IMREAD_UNCHANGED)

cap.release()
cv2.destroyAllWindows()
