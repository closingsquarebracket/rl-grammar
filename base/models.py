from base.base import BaseModel


class SentenceLengthModel(BaseModel):
    """Measures the length of a state."""

    def predict(self, state):
        return len(state)