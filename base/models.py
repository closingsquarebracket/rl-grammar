from base.base import BaseModel
import numpy as np

class SentenceLengthModel(BaseModel):
    """Measures the difference to a target length"""

    def __init__(self, target_length):
        super(SentenceLengthModel, self).__init__()
        self.property_size = 1
        self.target_length = target_length

    def predict(self, state):
        mismatch = self.target_length - len(state)
        reward = np.exp(- 3 * np.abs(mismatch)/self.target_length)
        return reward
