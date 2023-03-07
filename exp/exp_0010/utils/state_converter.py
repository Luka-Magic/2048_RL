import numpy as np
from gym.spaces import Box


class StateConverter:
    def __init__(self):
        # self.reference = np.tile(2**np.arange(16), (4, 4, 1)).transpose(2, 1, 0)
        # self.reference[0, :, :] = 0
        self.width = 4
        self.height = 4
    
    # def convert(self, state):
    #     new_obs = (self.reference == np.tile(state, (16, 1, 1))).astype(np.uint8)
    #     return new_obs
    
    def make_after_states(self, state):
        after_states = []
        can_actions = []
        scores = []
        copy_board = state.copy()
        for action in [0, 1, 2, 3]:
            rotated_obs = np.rot90(copy_board, k=action)
            score, updated_obs = self._slide_left_and_merge(rotated_obs)
            after_state = np.rot90(updated_obs, k=4 - action)
            if not updated_obs.all():
                can_actions.append(action)
                after_states.append(after_state)
                scores.append(score)
        return after_states, can_actions, scores
    
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