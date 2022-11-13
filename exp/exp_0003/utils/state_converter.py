import numpy as np
from gym.spaces import Box


class StateConverter:
    def __init__(self):
        self.reference = np.tile(2**np.arange(16), (4, 4, 1)).transpose(2, 1, 0)
        self.reference[0, :, :] = 0
        self.observation_spape = Box(
            low=0, high=255, shape=(16, 4, 4), dtype=np.uint8)
        self.width = 4
        self.height = 4
    
    def convert(self, state):
        new_obs = (self.reference == np.tile(state, (16, 1, 1))).astype(np.uint8)
        return new_obs
    
    def make_after_state(self, state, action):
        rotated_state = np.rot90(state, k=action)
        reward, updated_state = self._slide_left_and_merge(rotated_state)
        after_state = np.rot90(updated_state, k=4-action)
        return after_state, reward
    
    def _slide_left_and_merge(self, board):
        """Slide tiles on a grid to the left and merge."""

        result = []

        score = 0
        for row in board:
            row = np.extract(row > 0, row)
            score_, result_row = self._try_merge(row)
            score += score_
            row = np.pad(np.array(result_row), (0, self.width - len(result_row)),
                        'constant', constant_values=(0,))
            result.append(row)

        return score, np.array(result, dtype=np.int64)

    @staticmethod
    def _try_merge(row):
        score = 0
        result_row = []

        i = 1
        while i < len(row):
            if row[i] == row[i - 1]:
                score += row[i] + row[i - 1]
                result_row.append(row[i] + row[i - 1])
                i += 2
            else:
                result_row.append(row[i - 1])
                i += 1

        if i == len(row):
            result_row.append(row[i - 1])

        return score, result_row