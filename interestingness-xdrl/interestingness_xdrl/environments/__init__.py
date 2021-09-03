from abc import ABC, abstractmethod
from PIL import Image

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class EnvironmentData(object):
    """
    Represents a dataset of "external" information collected during the interaction of an agent with the environment.
    """

    def __init__(self, frames, observations, actions, new_episodes):
        """
        Creates a new environment dataset.
        :param list[Image.Image] frames: the visual representation of the environment at each timestep.
        :param list observations: the agent's observations at each timestep.
        :param list actions: the agent's actions at each timestep.
        :param list[int] new_episodes: a list with the timesteps in which new episodes started, which should include
        the first episode (`t=0`).
        """
        self.frames = frames
        self.observations = observations
        self.actions = actions
        self.new_episodes = new_episodes


class Environment(ABC):
    """
    An interface for environments (game engines, simulators) that agents can interact with.
    """

    def __init__(self, seed):
        """
        Creates a new simulation environment.
        :param int seed: the seed used to initialize the environment elements.
        """
        self.seed = seed

    @abstractmethod
    def collect_all_data(self, max_eps=0):
        """
        Collects all data from this environment by testing an agent for a specified amount of episodes.
        :param int max_eps: the maximum number of episodes to be run. `<=0` will run all episodes.
        :rtype: EnvironmentData
        :return: the data collected during the agent's interaction with the environment.
        """
        pass
