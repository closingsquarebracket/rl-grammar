from base.base import BaseModel
import numpy as np

class SentenceLengthModel(BaseModel):
    """Measures the difference to a target length"""

    def __init__(self, target_length, width: [None, int, float] = None):
        super(SentenceLengthModel, self).__init__()
        self.property_size = 1
        self.target_length = target_length
        if not width:
            self.scale_width = self.target_length / 10
        else:
            self.scale_width = width

    def predict(self, state):
        mismatch = self.target_length - len(state)
        reward = np.exp(- self.scale_width * np.abs(mismatch)/self.target_length)
        return reward
