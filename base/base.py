import numpy as np


class BaseEnvironment:
    """Base Environment for any environment.

    Implements the following functions:

     step(action), which translates the incoming action to an observation (the current
     state after the action), a reward (how successful that action was), done (a signal
     whether the game is finished) and an info, used for debugging.

     reset(), resetting the environment and all variables to their default values. It returns
     the initial observation for the environment.

     configure(config), which changes the environment according to config. It also returns
     the current configuration."""

    def __init__(self):
        self.current_state = None
        # The environment needs to keep track of some kind of state through-
        # out a game. The state is a representation of some underlying physical
        # process or alternatively a game state. It needs to be amendable through
        # actions in steps.
        self.current_reward = None
        # The reward refers to quality of the current state and relates to how
        # good the previous actions on the state were. The initial state should
        # be zero whereas an ideal state should be one.
        self.done = False
        # The done signal relates to the fact whether a game has finished.
        self.info = None  # info can be used for debugging purposes. It is passed on to the  # agent during stepping.

    def step(self, action: np.ndarray):
        """The action is the output of the agent and will usually be either a one-hot vector
        indicating a choice of action out of a set or an array of numbers indicating values.
        The action needs to be interpreted and enacted upon the current state. The result
        of the action is the observation that is fed back to the neural network. Corresponding
        to the observation is the reward, indicating the degree of success of the step. The
        starting reward should be 0 (when the game starts) and a successful end of the game
        should result in a reward of 1."""

        observation = self.current_state
        reward = self.current_reward
        done = self.done
        info = self.info

        return observation, reward, done, info

    def reset(self):
        observation = self.current_state
        return observation

    def configure(self, config):
        return config

    def debug(self, state = None, action = None):
        """The debug function can be used to simulate the state behaviour under some action.
        It should perform the same behaviour as self.step on a provided state, without disturbing
        the current state."""


class BaseModel:
    """A Model of a particular property of the model needs implement a predict function
    that translates from a current state to a property of that state. The size of the resulting
    properties (vector length) can be read by the surrounding environment via the property
    self.property_size."""

    def __init__(self):
        self.property_size = None

    def predict(self, state):
        """The predict function needs to take the current state of the environment and
        return a property (or several) thereof. The output size needs to match the property_size
        of the Model."""
        property_value = None
        return property_value


