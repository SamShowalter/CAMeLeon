import os
import logging
import pandas as pd
from collections import OrderedDict
from interestingness_xdrl.analysis.full import FullAnalysis

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

EPISODE_STR = 'Episode'
TIME_STEP_STR = 'Timestep'


class DatasetReport(object):
    """
    A report that saves a dataset on a CSV file containing labels for the different interestingness dimensions
    at each timestep of analysis.
    """

    def __init__(self, full_analysis, output_dir):
        """
        Creates a new interestingness dataset extractor.
        :param FullAnalysis full_analysis: the full introspection analysis over the agent's history of interaction
        with the environment. Has to contain valid (non-None) visual observations, otherwise an error will be raised.
        :param str output_dir: the path to the directory in which to save results.
        """
        self.full_analysis = full_analysis
        self.output_dir = output_dir
        self._data = full_analysis.data

    def create(self):
        logging.info('===================================================================')
        logging.info(f'Extracting dataset from {len(self._data)} interaction timesteps...')

        # add timesteps and episode data
        ep = -1
        ep_t = 0
        symbolic_data = OrderedDict({EPISODE_STR: [], TIME_STEP_STR: []})
        numerical_data = OrderedDict({EPISODE_STR: symbolic_data[EPISODE_STR],
                                      TIME_STEP_STR: symbolic_data[TIME_STEP_STR]})
        for datapoint in self._data:
            if datapoint.new_episode or ep == -1:  # make sure we increment ep in first step
                ep += 1
                ep_t = 0
            symbolic_data[EPISODE_STR].append(ep)
            symbolic_data[TIME_STEP_STR].append(ep_t)
            ep_t += 1

        # gets element at each time-step and add to list
        for level, dimension, analysis in self.full_analysis.elem_iterator:
            symbolic_data[dimension] = []
            numerical_data[dimension] = []
            for t in range(len(self._data)):
                elem, val = analysis.get_element_time(t)
                symbolic_data[dimension].append(elem)
                numerical_data[dimension].append(val)

        # save to csv files
        file_name = os.path.join(self.output_dir, 'int_elements_symb.csv')
        df = pd.DataFrame.from_dict(symbolic_data)
        df.to_csv(file_name, index=False)
        logging.info(f'Saved CSV file with symbolic information to:\n\t{file_name}.')

        file_name = os.path.join(self.output_dir, 'int_elements_num.csv')
        df = pd.DataFrame.from_dict(numerical_data)
        df.to_csv(file_name, index=False)
        logging.info(f'Saved CSV file with numerical information to:\n\t{file_name}.')
