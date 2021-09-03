import numbers
import os
import numpy as np
from scipy.spatial import distance
from pysc2.env.environment import TimeStep
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES, FeatureType
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.util.math import get_jensen_shannon_divergence
from interestingness_xdrl.util.plot import plot_evolution, plot_bar

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

OBS_FEAT_DIFFERENCES = 'Obs. Features Differences'
VALUE_DIFFERENCES = 'Value Differences'
ACTION_VALUE_DIFFERENCES = 'Action Value Differences'
POLICY_DIFFERENCES = 'Policy Differences'


def get_state_value_differences(datapoints1, datapoints2):
    """
    Gets the absolute differences between the state values of two sequences of datapoints.
    :param list[InteractionDataPoint] datapoints1: the first list of datapoints.
    :param list[InteractionDataPoint] datapoints2: the second list of datapoints.
    :rtype: np.ndarray
    :return: an array of shape (num_datapoints, ) containing the computed difference of values between each pair of
    provided datapoints.
    """
    return np.abs(np.array([datapoint.value for datapoint in datapoints1]) -
                  np.array([datapoint.value for datapoint in datapoints2]))


def get_action_probabilities_differences(datapoints1, datapoints2):
    """
    Gets the differences, computed as the JSD, between the action prob. distributions of two sequences of datapoints.
    :param list[InteractionDataPoint] datapoints1: the first list of datapoints.
    :param list[InteractionDataPoint] datapoints2: the second list of datapoints.
    :rtype: np.ndarray
    :return: an array of shape (num_action_factors, num_datapoints) containing the computed prob. dist. divergence for
    each action factor and pair of provided datapoints.
    """
    return np.array([[
        get_jensen_shannon_divergence(datapoints1[i].action_probs[a], datapoints2[i].action_probs[a])
        for i in range(len(datapoints1))]
        for a in range(len(datapoints1[0].action_factors))])


def get_action_values_differences(datapoints1, datapoints2):
    """
    Gets the differences, computed as the Euclidean distance, between the action values of two sequences of datapoints.
    :param list[InteractionDataPoint] datapoints1: the first list of datapoints.
    :param list[InteractionDataPoint] datapoints2: the second list of datapoints.
    :rtype: np.ndarray
    :return: an array of shape (num_action_factors, num_datapoints) containing the computed difference of values for
    each action factor and pair of provided datapoints.
    """
    return np.array([[
        np.linalg.norm(datapoints1[i].action_values[a] - datapoints2[i].action_values[a])
        for i in range(len(datapoints1))]
        for a in range(len(datapoints1[0].action_factors))])


def get_observation_differences(agent_obs1, agent_obs2):
    """
    Gets the difference between two SC2 agent observations. The differences are computed for each component of the
    observations and the metrics used depend on their type.
    :param TimeStep agent_obs1: the first agent observation.
    :param TimeStep agent_obs2: the second agent observation.
    :rtype: (int, float, float, dict)
    :return: a tuple containing the difference in step type, the difference in reward, the difference in discount,
    and a dictionary containing the differences for each observation component.
    """
    step_type_diff = 1 - int(agent_obs1.step_type == agent_obs2.step_type)
    reward_diff = np.abs(agent_obs1.reward - agent_obs2.reward)
    discount_diff = np.abs(agent_obs1.discount - agent_obs2.discount)
    obs_diffs = {}
    for k in agent_obs1.observation:
        if k == 'feature_screen' or k == 'feature_minimap':
            features = {f.name: f for f in (SCREEN_FEATURES if k == 'feature_screen' else MINIMAP_FEATURES)}
            obs_diffs[k] = {}
            for f in agent_obs1.observation[k]._index_names[0]:
                if features[f].type == FeatureType.CATEGORICAL:
                    # edit distance for categorical data
                    obs_diffs[k][f] = distance.hamming(
                        np.asarray(agent_obs1.observation[k][f]).flatten(),
                        np.asarray(agent_obs2.observation[k][f]).flatten())
                else:
                    # normalized RMSE for numerical features
                    obs_diffs[k][f] = np.sqrt(np.square(
                        np.asarray(agent_obs1.observation[k][f]) -
                        np.asarray(agent_obs2.observation[k][f])).mean()) / features[f].scale
        elif isinstance(agent_obs1.observation[k], str):
            obs_diffs[k] = 1 - int(agent_obs1.observation[k] == agent_obs2.observation[k])
        elif isinstance(agent_obs1.observation[k], numbers.Number):
            obs_diffs[k] = np.abs(agent_obs1.observation[k] - agent_obs2.observation[k])
        elif isinstance(agent_obs1.observation[k], np.ndarray):
            obs_diffs[k] = np.linalg.norm(np.asarray(agent_obs1.observation[k]) - np.asarray(agent_obs2.observation[k]))
        else:
            raise ValueError('Unknown type for feature: {} ({})'.format(k, type(agent_obs1.observation[k])))

    # return all diffs
    return step_type_diff, reward_diff, discount_diff, obs_diffs


def compare_datapoints(datapoints1, datapoints2, out_dir, name1='', name2='', screen_features=None):
    """
    Performs evaluation over the given pairs of datapoints and saves plots and CSV files with the comparisons.
    :param list[InteractionDataPoint] datapoints1: the first list of datapoints.
    :param list[InteractionDataPoint] datapoints2: the second list of datapoints.
    :param str out_dir: the path to the directory in which to save the results.
    :param str name1: the name of the first set of datapoints.
    :param str name2: the name of the second set of datapoints.
    :param list[str] screen_features: the names of the `feature_screen` layers for which to print a comparison.
    :rtype: dict[str, np.ndarray]
    :return: a dictionary containing the evaluation results of comparing the given datapoints.
    """
    evals = {}

    # obs difference
    differences = [get_observation_differences(datapoints1[i].observation, datapoints2[i].observation)
                   for i in range(len(datapoints1))]
    feat_differences = {}
    for f in (screen_features if screen_features is not None else [f.name for f in SCREEN_FEATURES]):
        feat_differences[f] = [diff[3]['feature_screen'][f] for diff in differences]
    plot_evolution(np.stack(feat_differences.values()), list(feat_differences.keys()),
                   '{} vs {} Observations'.format(name1, name2), output_img=os.path.join(out_dir, 'eval-obs.pdf'),
                   x_label='Time', y_label='Distance')
    plot_bar({k: [np.mean(feat_differences[k]), np.std(feat_differences[k]) / len(feat_differences[k])]
              for k in feat_differences}, 'Mean {} vs {} Observation Differences'.format(name1, name2),
             os.path.join(out_dir, 'eval-obs-mean.pdf'), plot_mean=False, y_label='Distance')
    evals.update({'{} ({})'.format(OBS_FEAT_DIFFERENCES, f): v for f, v in feat_differences.items()})

    # value difference
    real_values = np.array([real_datapoint.value for real_datapoint in datapoints1])
    vae_values = np.array([vae_datapoint.value for vae_datapoint in datapoints2])
    val_diffs = np.abs(real_values - vae_values)
    plot_evolution(np.stack((real_values, vae_values, val_diffs)),
                   [name1, name2, 'Abs. Diff.'], '{} vs {} Agent Values'.format(name1, name2),
                   output_img=os.path.join(out_dir, 'eval-value.pdf'), x_label='Time')
    evals[VALUE_DIFFERENCES] = val_diffs

    # policy diff
    act_probs_divs = get_action_probabilities_differences(datapoints1, datapoints2)
    plot_evolution(act_probs_divs, datapoints1[0].action_factors,
                   '{} vs {} Policy J-S Divergence'.format(name1, name2),
                   output_img=os.path.join(out_dir, 'eval-policy.pdf'), x_label='Time', y_label='JSD')
    plot_bar({a: [np.mean(act_probs_divs[i]), np.std(act_probs_divs[i]) / len(act_probs_divs[i])]
              for i, a in enumerate(datapoints1[0].action_factors)},
             'Mean {} vs {} Action Factor JSD'.format(name1, name2),
             os.path.join(out_dir, 'eval-policy-mean.pdf'), plot_mean=False, y_label='JSD')
    evals.update({'{} ({})'.format(POLICY_DIFFERENCES, a): act_probs_divs[i]
                  for i, a in enumerate(datapoints1[0].action_factors)})

    # action values
    act_vals_diffs = get_action_values_differences(datapoints1, datapoints2)
    plot_evolution(act_vals_diffs, datapoints1[0].action_factors,
                   '{} vs {} Action-Value Differences'.format(name1, name2),
                   output_img=os.path.join(out_dir, 'eval-action-value.pdf'),
                   x_label='Time', y_label='Action Value Diff.')
    plot_bar({a: [np.mean(act_vals_diffs[i]), np.std(act_vals_diffs[i]) / len(act_vals_diffs[i])]
              for i, a in enumerate(datapoints1[0].action_factors)},
             'Mean {} vs {} Action Factor Value Diff.'.format(name1, name2),
             os.path.join(out_dir, 'eval-action-value-mean.pdf'), plot_mean=False, y_label='Distance')
    evals.update({'{} ({})'.format(ACTION_VALUE_DIFFERENCES, a): act_vals_diffs[i]
                  for i, a in enumerate(datapoints1[0].action_factors)})

    return evals
