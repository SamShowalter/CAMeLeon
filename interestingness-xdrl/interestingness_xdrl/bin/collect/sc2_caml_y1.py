import json
import logging
import os
import  numpy as np
from absl import app, flags
from pysc2.lib.actions import ActionSpace
from interestingness_xdrl.environments.sc2 import SC2Environment
from interestingness_xdrl.util.io import create_clear_dir, save_object, get_file_name_without_extension
from interestingness_xdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Collects interaction data from CAML Y1 policies, learned using the **reaver** RL toolkit.'

flags.DEFINE_string('task_module', None, 'Fully-qualified name of importable module defining task-creation API.')
flags.DEFINE_string('policy', None, 'Policy spec; must be a symbol defined in the --task_module.')
flags.DEFINE_string('environment', None, 'Environment spec; must be a symbol defined in the --task_module.')
flags.DEFINE_bool('crop_to_playable_area', False, 'Argument to AgentInterfaceFormat')
flags.DEFINE_string('sc2data_root', None,
                    'Root directory for input data (such as Reaver checkpoints). Usually this'
                    ' should point to the \'sc2data\' repository, but it could be any directory'
                    ' that has the same subdirectory structure. You can also set this via the'
                    ' \'CAML_SC2DATA_ROOT\' environment variable; the flag will override the'
                    ' environment variable if both are present.')

flags.DEFINE_string('replay_sc2_version', None,
                    'SC2 version to use for replay. Either "x.y.z" or "latest". If not specified,'
                    ' version is inferred from the replay file. This ought to work, but if that'
                    ' specific version is missing (which seems to happen on Windows), it will'
                    ' raise an error.')

flags.DEFINE_string('vae_model', None, 'Path to the root directory containing the VAE model.')
flags.DEFINE_string('pe_model', None, 'Path to the root directory containing the Predictive Ensemble model.')
flags.DEFINE_bool('det_vae', True, 'Whether generated VAE obs are deterministic.')
flags.DEFINE_bool('det_pe', True, 'Whether predicted PE obs are deterministic.')
flags.DEFINE_integer('horizon', 16, 'Planning steps for prediction.')

flags.DEFINE_integer('batch_size', -1, 'The batch size used to sample the policy. `-1` uses all observations at once.')
flags.DEFINE_string('output', 'output/interaction_data', 'Path to the directory in which to save the collected data.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')
flags.DEFINE_integer('seed', 0, 'Random seed for both the agent\'s policy.')

flags.mark_flags_as_required(['task_module', 'policy', 'environment'])

RWD_DIM = 1


def main(unused_argv):
    args = flags.FLAGS

    # checks output dir and log file
    out_dir = os.path.join(args.output, get_file_name_without_extension(args.replays))
    create_clear_dir(out_dir, args.clear)
    change_log_handler(os.path.join(out_dir, 'data_collection.log'), args.verbosity)

    # save args
    with open(os.path.join(out_dir, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    # create environment and collect all data from the replay
    # (following AIF definitions in "sc2scenarios.scenarios.assault.spaces.caml_year1_eval.get_agent_interface_format")
    logging.info('Collecting all environment data from replay file: "{}"...'.format(args.replays))
    env = SC2Environment(args.replays, args.step_mul, args.replay_sc2_version,
                         feature_screen=(192, 144), feature_minimap=72, camera_width_world_units=48,
                         crop_to_playable_area=args.crop_to_playable_area, use_raw_units=True,
                         action_space=ActionSpace.RAW)
    env_data = env.collect_all_data()
    agent_obs = env_data.observations

    # create agent according to arguments
    if args.task_module is not None and \
            args.policy is not None and \
            args.environment is not None:
        from interestingness_xdrl.agents.sc2_caml_y1 import SC2CAMLY1Agent
        logging.info('Loading reaver agent...')

        agent = SC2CAMLY1Agent(
            env.agent_interface_format, args.seed, args.sc2data_root, args.task_module, args.environment, args.policy)

        logging.info('Collecting interaction data for {} steps...'.format(len(agent_obs)))
        dataset = agent.get_interaction_datapoints(agent_obs, args.batch_size)
    else:
        logging.info('Could not determine agent to load, missing arguments...')
        return

    # saves data
    data_file = os.path.join(out_dir, 'interaction_data.pkl.gz')
    logging.info('Saving results to\n\t"{}"...'.format(data_file))
    save_object(dataset, data_file)
    logging.info('Finished after {} timesteps ({} episodes)!'.format(
        len(dataset), len(np.where([dp.new_episode for dp in dataset])[0])))


if __name__ == '__main__':
    app.run(main)
