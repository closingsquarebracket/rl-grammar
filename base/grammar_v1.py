from collections import deque

import numpy as np

from base.base import BaseEnvironment

__author__ = """closingsquarebracket"""
__version__ = """1.0"""
__doc__ = """Grammar environment"""

class WordList:
    """Deque-based list for holding words in a row. Provides basic over-index protection."""
    def __init__(self):
        self.data = deque()
        self.position = 0

    def assign(self, vector):
        if self.position == -1:
            self.data.appendleft(vector)
            self.position += 1 # reset to zero position
        elif self.position == len(self.data) + 1:
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

    def __call__(self, *args, **kwargs):
        return list(self.data)

class GrammarEnvironment(BaseEnvironment):
    """Central environment for evaluating sentences."""

    def __init__(self, *models, change_steps = 200):
        """Models are instantiated outside of the Environment and passed as a BaseModel class instance.
        Models need to implement a .predict function. """
        super(GrammarEnvironment, self).__init__()
        self.models = models
        self.current_sentence = WordList()
        self.change_steps = 0
        self.max_changes = change_steps

    def action_to_state(self, choice, vector):
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
        return self.current_sentence() # convert to list


    def group_rewards(self, rewards: list) -> float:
        """TODO: Find a way to group the rewards of the models together to a single float value representing the state reward."""
        raise NotImplementedError

    def step(self, action: [list, dict]):
        """The action is expected to contain two components: A choice of action indicating whether a word is replaced,
        the position in the sentence is moved, or whether to end actions. The second element is a word vector to be used
        when doing a word replacement action.

        The action is one of four actions, indicated by a one-hot vector. The indicated actions are, in order:
        1: Replace current word (uses the given word-vector)
        2: Move cursor left (ignores given word-vector)
        3: Move cursor right (ignores given word-vector)
        4: Finish current game (ignores given word-vector)"""
        if type(action) == list:
            choice_of_action = action[0]
            word_vector = action[1]
        elif type(action) == dict:
            choice_of_action = action["choice"]
            word_vector = action["vector"]
        else:
            raise TypeError("The step in the GrammarEnvironment needs to be either a list or a dict (with 'action' and "
                            f"'vector' as possible entries). Received: {type(action)}")

        observation, _, done, info = super(GrammarEnvironment, self).step(action)

        self.current_state = self.action_to_state(choice_of_action, word_vector)

        observation = self.current_state

        rewards = [model.evaluate(self.current_state) for model in self.models]
        reward = self.group_rewards(rewards)

        self.change_steps += 1
        if self.change_steps >= self.max_changes:
            self.done = True
        return observation, reward, done, info


if __name__ == '__main__':
    from models import SentenceLenthModel
    sentence_length_model = SentenceLenthModel(30)
    grammar_environment = GrammarEnvironment(sentence_length_model)

