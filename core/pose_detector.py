import mediapipe as mp

mp_pose = mp.solutions.pose

class PoseDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False)

    def get_landmarks(self, frame):
        h, w, _ = frame.shape
        results = self.pose.process(frame[:, :, ::-1])

        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark

        points = {
            "ls": (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                   int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)),
            "rs": (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                   int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)),
            "lh": (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                   int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h)),
            "rh": (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                   int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h)),
        }

        return points
