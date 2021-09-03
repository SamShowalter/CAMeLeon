import jsonpickle

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class AnalysisConfiguration(object):
    """
    Represents a configuration for explanation analysis of deep reinforcement learning structures.
    """

    def __init__(self, num_processes=-1,
                 rwd_outlier_stds=2,
                 certain_exec_max_div=0.1, uncertain_exec_min_div=0.9,
                 value_outlier_stds=2., pred_error_outlier_stds=2.,
                 aleatoric_outlier_stds=2., epistemic_outlier_stds=2.
                 ):
        """
        Creates a new analysis configuration.
        :param int num_processes: the number of worker processes to use for evolution. If -1, then the number returned
        by `os.cpu_count()` is used.
        :param float rwd_outlier_stds: the number of standard deviations above/below which a point is considered an outlier.
        :param float certain_exec_max_div: the maximum diversity of action executions for a situation to be considered certain.
        :param float uncertain_exec_min_div: the minimum dispersion of action executions for a state to be considered uncertain.
        :param float value_outlier_stds: the threshold for the value of a state for it to be considered an outlier.
        :param float pred_error_outlier_stds: the threshold for the prediction error for a state to be considered an outlier.
        :param float aleatoric_outlier_stds: the number of standard deviations above/below which a point is considered an outlier.
        :param float epistemic_outlier_stds: the number of standard deviations above/below which a point is considered an outlier.
        """

        self.num_processes = num_processes

        # reward
        self.rwd_outlier_stds = rwd_outlier_stds

        # execution certainty
        self.certain_exec_max_div = certain_exec_max_div
        self.uncertain_exec_min_div = uncertain_exec_min_div

        # value
        self.value_outlier_stds = value_outlier_stds
        self.pred_error_outlier_stds = pred_error_outlier_stds

        # dynamics uncertainties
        self.aleatoric_outlier_stds = aleatoric_outlier_stds
        self.epistemic_outlier_stds = epistemic_outlier_stds


    def save_json(self, json_file_path):
        """
        Saves a text file representing this configuration in a JSON format.
        :param str json_file_path: the path to the JSON file in which to save this configuration.
        :return:
        """
        jsonpickle.set_preferred_backend('json')
        jsonpickle.set_encoder_options('json', indent=4, sort_keys=False)
        with open(json_file_path, 'w') as json_file:
            json_str = jsonpickle.encode(self)
            json_file.write(json_str)

    @classmethod
    def load_json(cls, json_file_path):
        """
        Loads an analysis object from the given JSON formatted file.
        :param str json_file_path: the path to the JSON file from which to load a configuration.
        :rtype: AnalysisConfiguration
        :return: the configuration object stored in the given JSON file.
        """
        with open(json_file_path) as json_file:
            return jsonpickle.decode(json_file.read())
