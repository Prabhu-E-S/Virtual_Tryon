class Stabilizer:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.prev = None

    def smooth(self, current):
        if self.prev is None:
            self.prev = current
            return current

        smoothed = {}
        for k in current:
            x = int(self.alpha * self.prev[k][0] + (1 - self.alpha) * current[k][0])
            y = int(self.alpha * self.prev[k][1] + (1 - self.alpha) * current[k][1])
            smoothed[k] = (x, y)

        self.prev = smoothed
        return smoothed
