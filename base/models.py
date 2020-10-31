import numpy as np

from base.base import BaseModel


class SentenceLength(BaseModel):
    """Measures the difference to a target length"""

    def __init__(self, target_length, width: [None, int, float] = None):
        super(SentenceLength, self).__init__()
        self.property_size = 1
        self.target_length = target_length
        if not width:
            self.scale_width = self.target_length / 10
        else:
            self.scale_width = width

    def predict(self, state) -> float:
        state, _ = state
        mismatch = self.target_length - len(state)
        reward = np.exp(- self.scale_width * np.abs(mismatch) / self.target_length)
        return reward


class WordVectorAccuracy(BaseModel):
    """Measures the difference between the current vector and the closest vector in the word vector space.
    This model provides no information regarding "correctness" of the current word."""

    def __init__(self, word_vectors: np.ndarray):
        super(WordVectorAccuracy, self).__init__()
        self.property_size = 1

        self.word_vectors = word_vectors
        # normalize all vectors for consistency
        self.word_vectors /= np.linalg.norm(self.word_vectors, axis = 1, keepdims = True)

    def predict(self, state) -> float:
        vectors, position = state
        current_vector = vectors[position]
        current_vector /= np.linalg.norm(current_vector, keepdims = True) # normalize before using
        accuracy = np.max(np.dot(self.word_vectors, current_vector))  # find closest match, then return the cos(angle)
        return accuracy # range: -1 to 1, with 1 ≘ full alignment, -1 ≘ antiparallel, 0 ≘ orthogonality.


class WordVectorAccuracyGensim(BaseModel):
    """Measures the difference between the current vector and the closest vector in the word vector space.
    Provides no information regarding correctness. Provides the same funtionality as WordVectorAccuracy, but requires
    the gensim package."""

    import gensim
    def __init__(self, keyed_vectors: gensim.models.keyedvectors.KeyedVectors):
        super(WordVectorAccuracyGensim, self).__init__()
        self.property_size = 1
        self.word_vectors = keyed_vectors

    def predict(self, state):
        vectors, position = state
        current_vector = vectors[position]
        [(_, accuracy)] = self.word_vectors.similar_by_vector(current_vector, topn = 1)
        return accuracy
