import os
from abc import ABC, abstractmethod
from interestingness_xdrl.analysis.full import FullAnalysis

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class CompetencyReport(ABC):
    """
    Defines a base class for methods of competency assessment and explanation, i.e., that provide reports or
    explanations based on given interaction data and the corresponding introspection analysis.
    """

    def __init__(self, full_analysis, output_dir):
        """
        Creates a new explainer.
        :param FullAnalysis full_analysis: the full introspection analysis over the agent's history of interaction
        with the environment.
        :param str output_dir: the path to the directory in which to save explanations / results.
        """
        self.output_dir = output_dir
        self.full_analysis = full_analysis

        # prepares directories
        for level, dimension, _ in full_analysis.elem_iterator:
            out_dir = os.path.join(self.output_dir, level, dimension)
            os.makedirs(out_dir, exist_ok=True)

    @abstractmethod
    def create(self):
        """
        Generate the competency assessment report.
        """
        pass
