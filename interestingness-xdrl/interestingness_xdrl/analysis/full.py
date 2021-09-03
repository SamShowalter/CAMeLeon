import os
import sys
import copy
import gzip
import logging
import pickle
from collections import OrderedDict, Iterable
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.analysis import AnalysisBase, AnalysisConfiguration
from interestingness_xdrl.analysis.interaction.action_value import ActionValueAnalysis
from interestingness_xdrl.analysis.interaction.execution_certainty import ExecutionCertaintyAnalysis
from interestingness_xdrl.analysis.interaction.execution_value import ExecutionValueAnalysis
from interestingness_xdrl.analysis.interaction.value import ValueAnalysis
from interestingness_xdrl.analysis.task.aleatoric_uncertainty import AleatoricUncertaintyAnalysis
from interestingness_xdrl.analysis.task.epistemic_uncert_jrd import EpistemicUncertaintyJRDAnalysis
from interestingness_xdrl.analysis.task.epistemic_uncert_kl_div import EpistemicUncertaintyKLDivAnalysis
from interestingness_xdrl.analysis.task.epistemic_uncert_var import EpistemicUncertaintyVarianceAnalysis
from interestingness_xdrl.analysis.task.reward import RewardAnalysis

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class FullAnalysis(AnalysisBase):
    """
    Represents a complete or full analysis, i.e., containing all possible analyses that can be performed.
    This is mostly useful to organize all analyses and create separate directories in which to save reports.
    """

    def __init__(self, data, analysis_config, analyses = ['all'],img_fmt='pdf'):
        """
        Creates a new analysis set.
        :param list[InteractionDataPoint] data: the interaction data collected to be analyzed.
        :param AnalysisConfiguration analysis_config: the analysis configuration containing the necessary parameters.
        :param str img_fmt: the format of the images to be saved.
        """
        super().__init__(data, analysis_config, img_fmt)
        self._analyses_dict = OrderedDict({
            '0-task': OrderedDict({
                'reward': RewardAnalysis,
                'aleatoric-uncertainty': AleatoricUncertaintyAnalysis,
                'epistemic-uncertainty-var': EpistemicUncertaintyVarianceAnalysis,
                'epistemic-uncertainty-jrd': EpistemicUncertaintyJRDAnalysis,
                'epistemic-uncertainty-kld': EpistemicUncertaintyKLDivAnalysis,
            }),
            '1-interaction': OrderedDict({
                'execution-uncertainty': ExecutionCertaintyAnalysis,
                'value': ValueAnalysis,
                'action-value': ActionValueAnalysis,
                'execution-value': ExecutionValueAnalysis,
            }),
            '2-meta': OrderedDict(
                {}),
        })

        if (isinstance(analyses,list) and (analyses[0] != 'all')):
            analyses_dict = copy.deepcopy(self._analyses_dict)
            analyses = set(analyses)
            for level, dim_analyses in self._analyses_dict.items():
                for dimension, analysis in dim_analyses.items():
                    if dimension not in analyses:
                        del analyses_dict[level][dimension]

        self._analyses_dict = analyses_dict
        self._analyses_list = []
        self.analysis_config = analysis_config
        self.img_fmt = img_fmt
        # for dim_analyses in self._analyses_dict.values():
        #     self._analyses_list.extend(dim_analyses.values())

    def __len__(self):
        return len(self._analyses_list)

    def __iter__(self):
        return iter(self._analyses_list)

    @property
    def elem_iterator(self):
        """
        Iterates over the analyses organized by level and dimension.
        :rtype: Iterable[str, str, AnalysisBase]
        :return: a 3-tuple containing the level name, analysis dimension name and analysis object.
        """
        for level, dim_analyses in self._analyses_dict.items():
            for dimension, analysis in dim_analyses.items():
                yield level, dimension, analysis

    def analyze(self, output_dir):
        # iterates analyses organized by level and dimension
        for level, dim_analyses in self._analyses_dict.items():
            logging.info('')
            logging.info('===================================================================')
            logging.info('Analyzing on introspection level \'{}\'...'.format(level))
            logging.info('===================================================================')
            self.analysis_config.out_root = os.path.join(output_dir,
                                       "{}_ep{}_rs{}_w{}".format(
                                           self.config.metadata['rollout']['rollout_env_name'],
                                           self.config.num_episodes,
                                           self.config.metadata['rollout']['seed'],
                                           self.config.metadata['rollout']['num_workers']))
            for dimension, analysis in dim_analyses.items():
                logging.info('___________________________________________________________________')
                self.analysis_config.outdir = os.path.join(self.analysis_config.out_root, level, dimension)
                os.makedirs(self.analysis_config.outdir, exist_ok=True)
                analysis = analysis(self.data,self.analysis_config, self.img_fmt, tag =dimension.replace("-","_"))
                self._analyses_list.append(analysis)
                self.data = analysis.analyze(self.analysis_config.outdir)


    def get_element_time(self, t):
        pass

    def get_element_datapoint(self, datapoint):
        pass

    def get_elements_time(self, t):
        """
        Gets the names of the interestingness elements identified by this analysis at the given simulation timestep
        and the associated interestingness values.
        :param int t: the simulation timestep in which to identify interestingness elements.
        :rtype: dict[str, float]
        :return: a dictionary containing the names and values of the interestingness elements at the given timestep.
        """
        elems = {}
        for analysis in self:
            name, value = analysis.get_element_time(t)
            elems[name] = value
        return elems

    def get_elements_datapoint(self, datapoint):
        """
        Analyzes a single interaction datapoint (as opposed to the whole history) and gets the name of the
        interestingness elements identified therein and the associated interestingness values.
        :param InteractionDataPoint datapoint: the interaction datapoint that we want to analyze.
        :rtype: dict[str, float]
        :return: a dictionary containing the names and values of the interestingness elements given the datapoint.
        """
        elems = {}
        for analysis in self:
            name, value = analysis.get_element_datapoint(datapoint)
            elems[name] = value
        return elems

    def save(self, file_path):
        """
        Saves a binary pickle file representing this object.
        :param str file_path: the path to the file in which to save this analysis.
        :return:
        """
        # avoids saving interaction data, saves only analysis data
        data = self.data
        for analysis in self._analyses_list + [self]:
            analysis.data = None

        with gzip.open(file_path, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

        for analysis in self._analyses_list + [self]:
            analysis.data = data

    @classmethod
    def load(cls, file_path, data=None):
        """
        Loads an analysis object from the given binary file.
        :param str file_path: the path to the binary file from which to load an object.
        :param InteractionDataset data: the interaction data to be set to all sub-analyses.
        :rtype: FullAnalysis
        :return: the object stored in the file.
        """
        with gzip.open(file_path, 'rb') as file:
            full_analysis = pickle.load(file)
            for analysis in list(full_analysis) + [full_analysis]:
                analysis.data = data
            return full_analysis
