import logging
import os
import csv
from feature_extractor.extractors import TIME_STEP_STR, EPISODE_STR
from interestingness_xdrl.reporting import CompetencyReport

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

# value for features meaning 'none of the other values'
DEFAULT_FEATURE_VAL = 'Undefined'


class SC2FeatureReport(CompetencyReport):
    """
    An explainer that extracts high-level features from PySC2 environments. In particular, it checks for changes in the
    values of features occurring when (or shortly before) the agent experiences an important situation, which is
    dictated by the occurrence of some extracted interestingness element.
    """

    def __init__(self, full_analysis, output_dir, features_file, before_timesteps, consistent_timesteps=3):
        """
        Creates a new high-level feature explainer.
        :param FullAnalysis full_analysis: the full introspection analysis over the agent's history of interaction
        with the environment.
        :param str output_dir: the path to the directory in which to save explanations / results.
        :param str features_file: the path to CSV file containing the extracted features.
        :param int before_timesteps: the number of environment time-steps to be analyzed for each element. `0` means
        that the features that are active at the timestep when an interestingness element was detected are output.
        Otherwise, the last `timesteps` are analyzed and features whose value changes during that period are output.
        """
        super().__init__(full_analysis, output_dir)
        self._before_timesteps = before_timesteps
        self.consistent_timesteps = consistent_timesteps

        # load features
        if not os.path.isfile(features_file):
            raise ValueError('Could not load features, file does not exist: {}'.format(features_file))
        self._feature_values = []
        with open(features_file) as fh:
            reader = csv.reader(fh, delimiter=',')

            # get feature names and verify needed features
            self._feature_names = next(reader)
            if TIME_STEP_STR not in self._feature_names or EPISODE_STR not in self._feature_names:
                raise ValueError('Feature dataset does not contain one or more necessary features: {}, {}'.format(
                    TIME_STEP_STR, EPISODE_STR))

            # reads rest of dataset
            self._feature_values.extend(reader)

        self._timesteps = len(self._feature_values)
        self._num_features = len(self._feature_names)
        self._timestep_idx = self._feature_names.index(TIME_STEP_STR)
        self._episode_idx = self._feature_names.index(EPISODE_STR)
        logging.info('Loaded {} features extracted for {} timesteps.'.format(self._num_features, self._timesteps))

    def create(self):
        logging.info('===================================================================')
        logging.info('Detecting changing features for {} preceding timesteps'.format(self._before_timesteps))

        ep_start = 0  # marks timestep when an episode starts
        ep_changed_features_idxs = []  # stores features that change in each episode
        elem_eps = {}  # stores features common to each element
        for t in range(self._timesteps):
            # check new episode (avoids going back to previous episode and capturing invalid feature changes)
            if self._feature_values[t][self._timestep_idx] == '0':
                if t > 0:
                    ep_changed_features_idxs.append(self._get_changed_features_idxs(ep_start, t - 1))
                ep_start = t
                logging.info('___________________________________________________________________')
                logging.info('New episode at {}...'.format(t))

            t_elems = []
            for level, dimension, analysis in self.full_analysis.elem_iterator:
                # gets element at this time-step and add to list
                elem, _ = analysis.get_element_time(t)
                if elem != '':
                    elem_full_name = os.path.join(self.output_dir, level, dimension, elem)
                    t_elems.append(elem_full_name)

            # ignore if no elements were captured at t
            if len(t_elems) == 0:
                continue

            # otherwise get features that changed and save a file for each element
            changed_features_idxs = self._get_changed_features_idxs(max(ep_start, t - self._before_timesteps), t)
            changed_features = self._get_features_labels(changed_features_idxs, t)
            for elem_full_name in t_elems:
                self._save_features(elem_full_name, changed_features, t)

                # also store the episode and timestep where this kind of element was detected
                if elem_full_name not in elem_eps:
                    elem_eps[elem_full_name] = []
                elem_eps[elem_full_name].append((len(ep_changed_features_idxs), t))

        # add changed features for last episode
        ep_changed_features_idxs.append(self._get_changed_features_idxs(ep_start, self._timesteps - 1))

        # save features that are active among all elements of a certain kind
        # considers only those that change in the episode
        for elem_full_name, eps_times in elem_eps.items():
            common_features = []
            for e, t in eps_times:
                active_changed_features = set(self._get_features_labels(ep_changed_features_idxs[e], t))
                common_features.append(active_changed_features)

            # saves intersection to file
            common_features_int = common_features[0]
            for i in range(1, len(common_features)):
                common_features_int.intersection_update(common_features[i])
            self._save_features(elem_full_name, common_features_int, 'common')

    def _get_changed_features_idxs(self, ep_start, t):
        # no time interval, just return all valid features at t
        if ep_start == t:
            changed_feats_idxs = set(range(self._num_features))
        else:
            # otherwise get indexes of features whose value changes in the given interval
            changed_feats_idxs = set()
            for f in range(self._num_features):
                if self._feature_values[ep_start][f] != self._feature_values[t][f]:
                    changed_feats_idxs.add(f)
        return changed_feats_idxs

    def _get_features_labels(self, feat_idxs, t):
        # return labels for the features in the given list that are also valid
        return [self._get_feature_label(f, feat)
                for f, feat in enumerate(self._feature_values[t])
                if f in feat_idxs and self._is_valid_feature(f, feat)]

    def _is_valid_feature(self, f, feat):
        # ignore metadata and 'none' features (more criteria can be added, eg a provided ignore list)
        return f != self._episode_idx and f != self._timestep_idx and feat != DEFAULT_FEATURE_VAL

    def _get_feature_label(self, f, feat):
        feature_name = self._feature_names[f].replace('_', ' ')
        if feat.lower() == 'true':
            # if feature is boolean and True, return the feature name
            return feature_name
        elif feat.lower() == 'false':
            # if feature is boolean and False, return 'not' the feature name
            return 'not {}'.format(feature_name)

        # otherwise return the feature name-value combination
        return '{}: {}'.format(feature_name, feat.replace('_', ' '))

    @staticmethod
    def _save_features(full_elem_name, changed_features, suffix):
        # writes a simple text file containing one feature label per line
        file_path = '{}-{}.txt'.format(full_elem_name, suffix)
        with open(file_path, 'w') as fh:
            for feature in sorted(changed_features):
                fh.write(feature + '\n')
        logging.info('Saved valid features for \'{}\' ({}) at: {}'.format(
            os.path.basename(full_elem_name), suffix, file_path))
