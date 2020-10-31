from collections import deque

import numpy as np

from base.base import BaseEnvironment

__author__ = """closingsquarebracket"""
__version__ = """1.0"""
__doc__ = """Grammar environment version 1. Simple model based on a list of words or word vectors. Models that provide
a reward are provided externally (base.models). """


class WordList:
    """Deque-based list for holding words in a row. Provides basic over-index protection. The position of the current
    index is tracked within the wordlist and can be extracted with WordList.position.

    The data structure is based on a collections.deque for fast modification on both ends. Data can be retrieved
    through WordList.data (type == deque) or through extract (type == list).

    Sample use case:
    >>>word_list = WordList()
    >>>word_list.left()
    >>>word_list.extract()
    ...[]
    >>>word_list.assign('a') # no type protection
    >>>word_list.assign('b')
    >>>word_list.extract()
    ...['b'] # assign does not move the index position
    >>>word_list.left()
    >>>word_list.assign('c')
    >>>word_list.extract()
    ...['c'. 'b']
    """

    def __init__(self):
        self.data = deque()
        self.position = 0

    def assign(self, vector):
        if self.position == -1:
            self.data.appendleft(vector)
            self.position += 1  # reset to zero position
        elif self.position == len(self.data) + 1:
            self.data.append(vector)
        else:
            if len(self.data) == 0:
                self.data.append(vector)
            else:
                self.data[self.position] = vector

    def left(self):
        self.position -= 1
        self.position = max(self.position, -1)

    def right(self):
        self.position += 1
        self.position = min(len(self.data) + 1, self.position)

    def __len__(self):
        return len(self.data)

    def extract(self):
        return list(self.data)

    def clear(self):
        self.data.clear()


class GrammarEnvironment(BaseEnvironment):
    """Central environment for evaluating sentences.

    Sentences are stored in a word_list instance."""

    def __init__(self, *models, change_steps = 200):
        """Models are instantiated outside of the Environment and passed as a BaseModel class instance.
        Models need to implement a .predict function. """
        super(GrammarEnvironment, self).__init__()
        self.models = models
        self.current_sentence = WordList()
        self.change_steps = 0
        self.max_changes = change_steps

    def action_to_state(self, choice, vector):
        """Converts an unpacked choice to a change on a WordList, using the vector where appropriate."""
        choice = int(np.argmax(choice))
        if choice == 0:
            self.current_sentence.assign(vector)
        elif choice == 1:
            self.current_sentence.left()
        elif choice == 2:
            self.current_sentence.right()
        elif choice == 3:
            self.done = True
        else:
            raise NotImplementedError("Only four types of actions available in GrammarEnvironment."
                                      f"Received: {choice} of type {type(choice)}."
                                      "Action needs to be encoded as one-hot.")
        return self.current_sentence

    def group_rewards(self, rewards: list) -> float:
        return sum(rewards) / len(rewards)  # simple average as a start

    def step(self, action: [list, dict]):
        """The action is expected to contain two components: A choice of action indicating whether a word is replaced,
        the position in the sentence is moved, or whether to end actions. The second element is a word vector to be used
        when doing a word replacement action.

        The action is one of four actions, indicated by a one-hot vector. The indicated actions are, in order:
        1: Replace current word (uses the given word-vector)
        2: Move cursor left (ignores given word-vector)
        3: Move cursor right (ignores given word-vector)
        4: Finish current game (ignores given word-vector)"""
        observation, reward, done, info = super(GrammarEnvironment, self).step(action)  # all outcomes need to replaced

        ### unpack incoming action
        if type(action) == list:
            choice_of_action = action[0]
            word_vector = action[1]
        elif type(action) == dict:
            choice_of_action = action["choice"]
            word_vector = action["vector"]
        else:
            raise TypeError("The step in the GrammarEnvironment needs to be either a list or a dict (with 'action' and "
                            f"'vector' as possible entries). Received: {type(action)}")

        ### perform action and change state
        self.current_state = self.action_to_state(choice_of_action, word_vector)

        ### prepare observation
        vectors = self.current_state.extract()  # convert into a list
        word_length = len(self.current_state)
        position = self.current_state.position
        one_hot_position = one_hot(word_length, position)
        observation = [vectors, one_hot_position]

        ### prepare rewards
        rewards = [model.predict([vectors, position]) for model in self.models]
        reward = self.group_rewards(rewards)

        ### adjust step counter
        self.change_steps += 1
        if self.change_steps >= self.max_changes:
            self.done = True

        ### retrieve done value from the action
        done = self.done

        ### provide basic debug output
        info = {'current step': self.change_steps}
        return observation, reward, done, info


def one_hot(length, position) -> np.ndarray:
    vector = np.zeros((1, length))  # shape in preparation for concatenation
    vector[position] = 1
    return vector


if __name__ == '__main__':
    from base.models import SentenceLength

    sentence_length_model = SentenceLength(30)
    grammar_environment = GrammarEnvironment(sentence_length_model)
