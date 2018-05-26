from collections import deque

import numpy as np


class Heatmap:
    """Class for storing heatmaps from the previous frames

    """
    def __init__(self, n_last_heatmaps=3):
        """init

        :param n_last_heatmaps: size of frame memory
        """
        self.queue = deque(maxlen=n_last_heatmaps)

    def sum_heatmap(self):
        """Sum all stored heatmaps

        :return: total heatmap as numpy array
        """
        return np.array(self.queue).sum(axis=0)