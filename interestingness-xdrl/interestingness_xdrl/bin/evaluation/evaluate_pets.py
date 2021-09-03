import json
import logging
import os
import numpy as np
import reaver.run2  # needed to load necessary reaver arguments
from absl import app, flags
from pysc2.lib import point_flag
from imago.models.semframe import model_remapper
from imago.models.sequential.pets.converters.reaver_vae_converter import ReaverVAESampleConverter
from imago.models.sequential.pets.bnn import BNN, setup_tf
from interestingness_xdrl.prediction import get_agent_rollouts
from interestingness_xdrl.agents.sc2_reaver import SC2ReaverAgent
from interestingness_xdrl.environments.sc2 import SC2Environment
from interestingness_xdrl.evaluation import compare_datapoints, get_observation_differences
from interestingness_xdrl.util.image import get_mean_image, get_variance_heatmap
from interestingness_xdrl.util.io import create_clear_dir, get_files_with_extension, get_file_name_without_extension
from interestingness_xdrl.util.logging import change_log_handler
from interestingness_xdrl.util.plot import plot_bar
from interestingness_xdrl.util.video import save_video

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Tests Predictive Ensemble model on given SC2 replays.'

FEATURE_NAMES = ['feature_screen', 'feature_minimap', 'available_actions', 'player']

flags.DEFINE_string('env', None, 'Environment in which to run the agent. Can be the Gym env id, PySC2 map, etc.')
flags.DEFINE_string('replay_sc2_version', None,
                    'SC2 version to use for replay. Either "x.y.z" or "latest". If not specified,'
                    ' version is inferred from the replay file. This ought to work, but if that'
                    ' specific version is missing (which seems to happen on Windows), it will'
                    ' raise an error.')

flags.DEFINE_string('vae_model', None, 'Path to the root directory containing the VAE model.')
flags.DEFINE_string('pe_model', None, 'Path to the root directory containing the Predictive Ensemble model.')
flags.DEFINE_multi_integer('rollout_lengths', [1], 'Length(s) of the predictive ensemble rollouts to evaluate.')
flags.DEFINE_integer('max_eps', 0, 'Maximum number of episodes from which to collect data.')
flags.DEFINE_integer('batch_size', 32, 'Size of batches to sample the predictive model.')
flags.DEFINE_bool('det_vae', True, 'Whether generated VAE obs are deterministic.')
flags.DEFINE_bool('det_pe', True, 'Whether predicted PE obs are deterministic.')

flags.DEFINE_string('replays', None, 'Path to replay file(s).')
flags.DEFINE_string('output', 'output', 'Path to the directory in which to save the collected data.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')

flags.DEFINE_integer('video_eps', 0,
                     'Num. episodes for which to save videos of the screen, original and reconstructed SC2 frames.')
flags.DEFINE_float('fps', 22.5, 'The frames per second ratio used to save the videos.')
flags.DEFINE_integer('crf', 18, 'Video constant rate factor: the default quality setting in `[0, 51]`')

point_flag.DEFINE_point('window_size', '1024,768', 'SC2 window size.')
flags.DEFINE_bool('hide_hud', True, 'Whether to hide the HUD / information panel at the bottom of the screen.')

flags.mark_flags_as_required(['replays', 'experiment', 'env', 'vae_model', 'pe_model'])

RWD_DIM = 1


def process_replay(args, file, vae_model, session):
    logging.info('Replaying {}...'.format(file))

    # create environment
    env = SC2Environment(file, args.step_mul, 1., args.replay_sc2_version,
                         window_size=args.window_size,
                         hide_hud=args.hide_hud,
                         capture_screen=args.video_eps > 0,
                         feature_screen=args.obs_spatial_dim,
                         use_feature_units=True)

    # collect all data from the replay
    env_data = env.collect_all_data(args.max_eps)
    screen_buffer = env_data.frames
    agent_obs = env_data.observations
    agent_actions = env_data.actions
    max_video_step = env_data.new_episodes[args.video_eps]

    out_dir = os.path.join(args.output, get_file_name_without_extension(file))
    create_clear_dir(out_dir, args.clear)
    if args.video_eps > 0:
        file_path = os.path.join(out_dir, '{}-screen.mp4'.format(get_file_name_without_extension(file)))
        save_video(screen_buffer[:max_video_step], file_path, args.fps, args.crf)

    # create agent
    agent = SC2ReaverAgent(
        env.agent_interface_format, args.seed, args.results_dir, args.experiment, args.env,
        args.obs_features, args.action_set, args.action_spatial_dim,
        args.safety, args.safe_distance, args.safe_flee_angle_range, args.safe_margin, args.safe_unsafe_steps,
        args.gin_files, args.gin_bindings, args.agent, args.gpu)

    # create converter from/to agent observations using VAE
    converter = ReaverVAESampleConverter(env.agent_interface_format, vae_model, args.action_set,
                                         args.action_spatial_dim, args.det_vae)

    # collects real observation images and saves video
    real_ag_obs_img_buffer = converter.to_images(agent_obs[:max_video_step])
    file_path = os.path.join(out_dir, '{}-orig-layers.mp4'.format(get_file_name_without_extension(file)))
    save_video(real_ag_obs_img_buffer, file_path, args.fps, args.crf, verify_input=True, color='white')

    # gets predictive rollouts of length l for each step and using the agent's policy
    pe_model = BNN.load(args.pe_model, converter.observation_dim, converter.action_dim, RWD_DIM, session)

    evals = []
    for length in args.rollout_lengths:
        out_dir_ = os.path.join(out_dir, str(length))
        create_clear_dir(out_dir_, args.clear)

        logging.info('Evaluating PE for rollouts of length {}, saving results to\n\t"{}"'.format(length, out_dir_))

        obs_rollouts, rwd_rollouts = get_agent_rollouts(pe_model, agent, converter, agent_obs, agent_actions,
                                                        RWD_DIM, args.batch_size, length, args.det_pe)

        # converts last latent observation as predicted by rollouts of each predictor to an agent observation, using vae
        rollout_rewards = [rwd_rollouts[-1, 0, i, :, :] for i in range(pe_model.num_nets)]
        rollout_agent_obs = [converter.to_agent_observations(
            obs_rollouts[-1, 0, i, :, :],
            rollout_rewards[i],
            agent_obs[length:] + agent_obs[-length:])
            for i in range(pe_model.num_nets)]

        # gets time-aligned predicted observations
        pred_agent_obs = [agent_obs[:length] + rollout_agent_obs[k][:-length]
                          for k in range(pe_model.num_nets)]

        # calculate mean obs diff at each timestep for each predictor
        real_datapoints = agent.get_interaction_datapoints(agent_obs)
        pred_datapoints = [agent.get_interaction_datapoints(pred_agent_obs[k])
                           for k in range(pe_model.num_nets)]
        mean_diffs = np.array([[
            np.mean(list(get_observation_differences(real_datapoints[i].observation,
                                                     pred_datapoints[k][i].observation)[3]['feature_screen'].values()))
            for i in range(len(real_datapoints))]
            for k in range(pe_model.num_nets)])

        # selects best predictor (lower mean diff) at each time step, save video
        best_idxs = np.argmin(mean_diffs, axis=0)
        best_pred_ag_obs = [pred_agent_obs[best_idxs[i]][i] for i in range(len(best_idxs))]
        pred_ag_obs_img_buffer = converter.to_images(best_pred_ag_obs[:max_video_step])
        file_path = os.path.join(out_dir_, '{}-best-pred-layers.mp4'.format(get_file_name_without_extension(file)))
        save_video(pred_ag_obs_img_buffer, file_path, args.fps, args.crf, verify_input=True, color='white')

        # collects predicted observations images (for each predictor) and saves video
        pred_ag_obs_img_buffers = []
        for k in range(pe_model.num_nets):
            # first 'rollout-length' observations cannot be predicted, so just copy original ones
            pred_ag_obs_img_buffer = converter.to_images(pred_agent_obs[k][:max_video_step])
            file_path = os.path.join(out_dir_, '{}-pred-layers-net-{}.mp4'.format(
                get_file_name_without_extension(file), k))
            save_video(pred_ag_obs_img_buffer, file_path, args.fps, args.crf, verify_input=True, color='white')
            pred_ag_obs_img_buffers.append(pred_ag_obs_img_buffer)

        # saves the mean and variance between layers
        pred_ag_obs_img_buffer = [get_mean_image(
            [pred_ag_obs_img_buffers[k][i] for k in range(pe_model.num_nets)], canvas_color='white')
            for i in range(len(pred_ag_obs_img_buffers[0]))]
        file_path = os.path.join(out_dir_, '{}-mean-pred-layers.mp4'.format(get_file_name_without_extension(file)))
        save_video(pred_ag_obs_img_buffer, file_path, args.fps, args.crf, verify_input=True, color='white')

        pred_ag_obs_img_buffer = [get_variance_heatmap(
            [pred_ag_obs_img_buffers[k][i] for k in range(pe_model.num_nets)], False, True, canvas_color='white')
            for i in range(len(pred_ag_obs_img_buffers[0]))]
        file_path = os.path.join(out_dir_, '{}-var-pred-layers.mp4'.format(get_file_name_without_extension(file)))
        save_video(pred_ag_obs_img_buffer, file_path, args.fps, args.crf, verify_input=True, color='white')

        # evaluates real vs predicted performance for each predictor
        length_evals = []
        for k in range(pe_model.num_nets):
            out_dir__ = os.path.join(out_dir_, 'net-{}-eval'.format(k))
            create_clear_dir(out_dir__, args.clear)
            length_evals.append(compare_datapoints(real_datapoints, pred_datapoints[k], out_dir__, 'Real',
                                                   'Predictor {} ($\\mathbf{{h={}}}$)'.format(k, length),
                                                   [c.name for c in vae_model.components]))
        evals.append(length_evals)

    # gets mean differences and errors from evaluation data
    rollout_diffs = {}
    for i, length_evals in enumerate(evals):  # for each rollout
        nets_data = {}
        for net_evals in length_evals:  # for each network in the ensemble
            # collect eval data for each network and eval criterion
            for eval_name in net_evals.keys():
                eval_data = net_evals[eval_name]
                if eval_name not in nets_data:
                    nets_data[eval_name] = []
                nets_data[eval_name].append(eval_data)

        # gets mean and standard error (across networks in ensemble) for each eval criterion
        if i == 0:
            rollout_diffs = {eval_name: {} for eval_name in nets_data.keys()}
        for eval_name, eval_data in nets_data.items():
            rollout_diffs[eval_name][str(args.rollout_lengths[i])] = [
                np.mean(eval_data), np.std(eval_data) / len(eval_data)]

    # saves data per rollout
    out_dir = os.path.join(out_dir, 'mean-eval')
    create_clear_dir(out_dir, args.clear)
    for eval_name, diffs in rollout_diffs.items():
        plot_bar(diffs, 'Mean {}'.format(eval_name),
                 os.path.join(out_dir, 'mean-{}.pdf'.format(eval_name.lower().replace(' ', '-').replace('.', ''))),
                 plot_mean=False, x_label='Rollout Length')


def main(unused_argv):
    args = flags.FLAGS

    # checks output dir and log file
    create_clear_dir(args.output, args.clear)
    change_log_handler(os.path.join(args.output, 'evaluate_pe.log'), args.verbosity)

    # save args
    with open(os.path.join(args.output, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    # setup TF
    session, has_gpu = setup_tf()

    # loads models
    vae_model = model_remapper.load_model(args.vae_model, device=None if has_gpu else 'cpu')

    # checks input files
    if os.path.isfile(args.replays):
        files = [args.replays]
    elif os.path.isdir(args.replays):
        files = list(get_files_with_extension(args.replays, 'SC2Replay'))
    else:
        raise ValueError('Input path is not a valid file or directory: {}.'.format(args.input))

    # process files
    for file in files:
        process_replay(args, file, vae_model, session)


if __name__ == '__main__':
    app.run(main)
