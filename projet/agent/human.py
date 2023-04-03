import numpy as np


class Human:
    def __init__(self):
        pass

    def get_action(self, mask):
        action = ""
        available_actions = np.flatnonzero(mask["action_mask"])
        while not action.isdigit() or int(action) not in available_actions:
            action = input("Command: ")
        return np.int64(action)

    def load(self):
        pass