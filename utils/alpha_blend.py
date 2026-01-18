import numpy as np

def alpha_blend(frame, overlay):
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        frame[:, :, c] = frame[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
    return frame
