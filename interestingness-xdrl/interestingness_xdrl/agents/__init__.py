from abc import ABC, abstractmethod
from interestingness_xdrl import InteractionDataPoint

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class Agent(ABC):
    """
    Represents an agent that interacts with an environment by executing actions, making observations, collecting data.
    """

    def __init__(self, seed):
        """
        Creates a new agent.
        :param int seed: the seed used to initialize the agent.
        """
        self.seed = seed

    @abstractmethod
    def get_counterfactuals(self, datapoint):
        """
        Gets interaction data for counterfactuals of the given datapoint. Counterfactuals are alternatives
        (similar/close) to the given datapoint's state/observation that potentially can lead to different behavior.
        :param InteractionDataPoint datapoint: the datapoint for which to compute counterfactuals.
        :rtype: list[tuple[InteractionDataPoint, str]]
        :return: a list of datapoints counterfactual to the given datapoint and corresponding description.
        """
        pass

    @abstractmethod
    def get_interaction_datapoints(self, observations):
        """
        Gets interaction datapoints given input observations summarizing the state of the agent's interaction with the
        environment.
        :param list observations: the agent's observations containing different observation features.
        :rtype: list[InteractionDataPoint]
        :return: a list with the agent's interaction datapoints given the observations.
        """
        pass
