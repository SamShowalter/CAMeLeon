import json
import logging
import os
import reaver.run2  # needed to load necessary reaver arguments
from absl import app, flags
from interestingness_xdrl.environments.sc2 import SC2Environment
from interestingness_xdrl.util.io import create_clear_dir, save_object, get_file_name_without_extension
from interestingness_xdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Collects interaction data from policies learned using the **reaver** RL toolkit.'

flags.DEFINE_string('env', None, 'Environment in which to run the agent. Can be the Gym env id, PySC2 map, etc.')
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

flags.DEFINE_string('output', 'output/interaction_data', 'Path to the directory in which to save the collected data.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')

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
    logging.info('Collecting all environment data from replay file: "{}"...'.format(args.replays))
    env = SC2Environment(args.replays, args.step_mul, args.replay_sc2_version,
                         feature_screen=args.obs_spatial_dim, feature_minimap=args.obs_spatial_dim,
                         use_feature_units=True)
    env_data = env.collect_all_data()
    agent_obs = env_data.observations
    agent_actions = env_data.actions

    # create agent according to arguments
    if args.experiment is not None and \
            args.env is not None and \
            args.vae_model is not None and \
            args.pe_model is not None:
        from imago.models.sequential.pets.bnn import setup_tf, BNN
        from imago.models.sequential.pets.converters.reaver_vae_converter import ReaverVAESampleConverter
        from imago.models.semframe import model_remapper
        from interestingness_xdrl.agents.sc2_reaver_vae_bnn import SC2ReaverVAEBNNAgent
        logging.info('Loading reaver agent with predictive model ensemble via VAE...')

        # setup TF
        session, has_gpu = setup_tf()

        # loads VAE
        rb_perturb_model = model_remapper.load_model(args.vae_model, device=None if has_gpu else 'cpu')

        # create converter from/to agent observations using VAE
        converter = ReaverVAESampleConverter(env.agent_interface_format, rb_perturb_model, args.action_set,
                                             args.action_spatial_dim, args.det_vae)

        # loads probabilistic ensemble of predictive models
        pe_model = BNN.load(args.pe_model, converter.observation_dim, converter.action_dim, RWD_DIM, session)

        agent = SC2ReaverVAEBNNAgent(
            converter, pe_model,
            env.agent_interface_format, args.seed, args.results_dir, args.experiment, args.env,
            args.obs_features, args.action_set, args.action_spatial_dim,
            args.safety, args.safe_distance, args.safe_flee_angle_range, args.safe_margin, args.safe_unsafe_steps,
            args.gin_files, args.gin_bindings, args.agent, args.gpu, args.horizon, args.det_pe)

        logging.info('Collecting interaction data for {} steps...'.format(len(agent_obs)))
        dataset = agent.get_interaction_datapoints((agent_obs, agent_actions))

    elif args.vae_model is not None and args.pe_model is not None:
        from imago.models.sequential.pets.converters.rb_vae_converter import RBVAESampleConverter
        from imago.models.behav.rb_perturb import RBPerturbModel
        from imago.models.sequential.pets.bnn import setup_tf, BNN
        from interestingness_xdrl.agents.sc2_rb_vae_bnn import SC2RBVAEBNNAgent

        logging.info('Loading agent with predictive model ensemble and reaver-like behavior via VAE...')

        # setup TF
        session, has_gpu = setup_tf()

        # loads VAE
        rb_perturb_model = RBPerturbModel(args.vae_model, 'GPU:0' if has_gpu else 'cpu')

        # create converter from/to agent observations using VAE
        converter = RBVAESampleConverter(env.agent_interface_format, rb_perturb_model, args.action_set,
                                         args.action_spatial_dim, args.det_vae)

        # loads probabilistic ensemble of predictive models
        pe_model = BNN.load(args.pe_model, converter.observation_dim, converter.action_dim, RWD_DIM, session)

        agent = SC2RBVAEBNNAgent(converter, pe_model, args.seed, args.horizon, args.det_pe)

        logging.info('Collecting interaction data for {} steps...'.format(len(agent_obs)))
        dataset = agent.get_interaction_datapoints(agent_obs)

    elif args.experiment is not None and args.env is not None:
        from interestingness_xdrl.agents.sc2_reaver import SC2ReaverAgent
        logging.info('Loading reaver agent...')

        agent = SC2ReaverAgent(
            env.agent_interface_format, args.seed, args.results_dir, args.experiment, args.env,
            args.obs_features, args.action_set, args.action_spatial_dim,
            args.safety, args.safe_distance, args.safe_flee_angle_range, args.safe_margin, args.safe_unsafe_steps,
            args.gin_files, args.gin_bindings, args.agent, args.gpu)

        logging.info('Collecting interaction data for {} steps...'.format(len(agent_obs)))
        dataset = agent.get_interaction_datapoints(agent_obs)
    else:
        logging.info('Could not determine agent to load, missing arguments...')
        return

    # saves data
    data_file = os.path.join(out_dir, 'interaction_data.pkl.gz')
    save_object(dataset, data_file)
    logging.info('Finished after {} timesteps, saved results to\n\t"{}"'.format(len(dataset), data_file))


if __name__ == '__main__':
    app.run(main)
