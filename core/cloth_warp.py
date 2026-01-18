import cv2
import numpy as np

def warp_cloth(cloth, dst_points, frame_shape):
    h, w = cloth.shape[:2]

    src_points = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    dst_points = np.float32(dst_points)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(cloth, matrix,
                                 (frame_shape[1], frame_shape[0]),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    return warped
