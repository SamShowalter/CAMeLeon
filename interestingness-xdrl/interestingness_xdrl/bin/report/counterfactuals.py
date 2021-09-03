import json
import logging
import os
import platform
import reaver.run2  # needed to load necessary reaver arguments
from absl import app, flags
from pysc2.lib import point_flag
from interestingness_xdrl.analysis.full import FullAnalysis
from interestingness_xdrl.reporting.counterfactuals import CounterfactualsReport
from interestingness_xdrl.environments.sc2 import SC2Environment
from interestingness_xdrl.util.io import create_clear_dir, load_object, get_file_name_without_extension
from interestingness_xdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Extracts counterfactual state examples, i.e., specific states that, if experienced by the agent, ' \
                  'would lead to a very different situation with regards to interestingness analysis.'

flags.DEFINE_string('env', None, 'Environment in which to run the agent. Can be the Gym env id, PySC2 map, etc.')
flags.DEFINE_string('replay_sc2_version', None,
                    'SC2 version to use for replay. Either "x.y.z" or "latest". If not specified,'
                    ' version is inferred from the replay file. This ought to work, but if that'
                    ' specific version is missing (which seems to happen on Windows), it will'
                    ' raise an error.')

flags.DEFINE_string('vae_model', None, 'Path to the root directory containing the VAE model.')
flags.DEFINE_bool('det_vae', True, 'Whether generated VAE obs are deterministic.')

flags.DEFINE_string('data', None,
                    'Pickle file containing the interaction data collected using `collect_data_*` scripts.')
flags.DEFINE_string('analysis', None,
                    'Pickle file containing the full introspection analysis collected using the `analyze` script.')

flags.DEFINE_bool('hide_hud', True, 'Whether to hide the HUD / information panel at the bottom of the screen.')
flags.DEFINE_string('output', 'output/report-counterfactuals',
                    'Path to the directory in which to save the collected data.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')


def main(unused_argv):
    args = flags.FLAGS

    # check for mac OS
    if platform.system() != 'Darwin':
        raise ValueError('Counterfactual extraction is currently supported only in non-macOS platform.')

    # checks output dir and log file
    out_dir = os.path.join(args.output, get_file_name_without_extension(args.replays))
    create_clear_dir(out_dir, args.clear)
    change_log_handler(os.path.join(out_dir, 'counterfactuals.log'), args.verbosity)

    # save args
    with open(os.path.join(out_dir, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    # load data file
    if not os.path.isfile(args.data):
        raise ValueError('Could not find interaction data file in {}'.format(args.data))
    interaction_data = load_object(args.data)
    logging.info('Loaded interaction data corresponding to {} timesteps from: {}'.format(
        len(interaction_data), args.data))

    # load full analysis
    if not os.path.isfile(args.analysis):
        raise ValueError('Could not find full analysis data file in {}'.format(args.analysis))
    analyses = FullAnalysis.load(args.analysis, interaction_data)
    logging.info('Loaded full analysis data file from: {}'.format(args.analysis))

    # create environment
    env = SC2Environment(args.replays, args.step_mul, 1., args.replay_sc2_version, 1, True,
                         args.window_size, args.hide_hud, True, args.obs_spatial_dim,
                         use_feature_units=True)
    env_data = env.collect_all_data()

    # create agent according to arguments
    if args.experiment is not None and \
            args.env is not None:
        from interestingness_xdrl.agents.sc2_reaver import SC2ReaverAgent
        logging.info('Loading reaver agent...')

        # create agent
        agent = SC2ReaverAgent(
            env.agent_interface_format, args.seed, args.results_dir, args.experiment, args.env,
            args.obs_features, args.action_set, args.action_spatial_dim,
            args.safety, args.safe_distance, args.safe_flee_angle_range, args.safe_margin, args.safe_unsafe_steps,
            args.gin_files, args.gin_bindings, args.agent, args.gpu)

    elif args.vae_model is not None:
        from imago.models.sequential.pets.converters.rb_vae_converter import RBVAESampleConverter
        from imago.models.behav.rb_perturb import RBPerturbModel
        from imago.models.sequential.pets.bnn import setup_tf
        from interestingness_xdrl.agents.sc2_rb_vae_bnn import SC2RBVAEBNNAgent

        logging.info('Loading agent with predictive model ensemble and reaver-like behavior via VAE...')

        # setup TF
        session, has_gpu = setup_tf()

        # create converter from/to agent observations using VAE
        rb_perturb_model = RBPerturbModel(args.vae_model, 'GPU:0' if has_gpu else 'cpu')
        converter = RBVAESampleConverter(env.agent_interface_format, rb_perturb_model, args.action_set,
                                         args.action_spatial_dim, args.det_vae)
        agent = SC2RBVAEBNNAgent(converter, None, args.seed)

    else:
        logging.info('Could not determine agent to load, missing arguments...')
        return

    # collects and saves counterfactuals
    explainer = CounterfactualsReport(analyses, out_dir, env_data.frames, agent)
    explainer.create()

    logging.info('Finished after {} timesteps!'.format(len(env_data.frames)))


if __name__ == '__main__':
    app.run(main)
