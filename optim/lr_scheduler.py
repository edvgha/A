from typing import List


class LearningRate:
    def __init__(self, params: List) -> None:
        self.type = params[0]
        if self.type == 'constant':
            self.constant_lr = params[1]

    def get(self) -> float:
        if self.type == 'constant':
            return self.constant_lr
        assert False

    def update(self, step: int):
        pass

    

