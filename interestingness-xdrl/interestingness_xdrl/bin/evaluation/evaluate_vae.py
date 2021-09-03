import json
import logging
import os
import numpy as np
import reaver.run2  # needed to load necessary reaver arguments
from absl import app, flags
from pysc2.lib import point_flag
from imago.models.semframe import model_remapper
from imago.models.sequential.pets.converters.reaver_vae_converter import ReaverVAESampleConverter
from interestingness_xdrl.agents.sc2_reaver import SC2ReaverAgent
from interestingness_xdrl.agents.sc2_reaver_vae import SC2ReaverVAEAgent
from interestingness_xdrl.evaluation import compare_datapoints
from interestingness_xdrl.environments.sc2 import SC2Environment
from interestingness_xdrl.util.io import create_clear_dir, get_files_with_extension, get_file_name_without_extension
from interestingness_xdrl.util.logging import change_log_handler
from interestingness_xdrl.util.plot import plot_evolution
from interestingness_xdrl.util.video import save_video

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Tests ConvVAE on given SC2 replays.'

FEATURE_NAMES = ['feature_screen', 'feature_minimap', 'available_actions', 'player']

flags.DEFINE_string('env', None, 'Environment in which to run the agent. Can be the Gym env id, PySC2 map, etc.')
flags.DEFINE_string('replay_sc2_version', None,
                    'SC2 version to use for replay. Either "x.y.z" or "latest". If not specified,'
                    ' version is inferred from the replay file. This ought to work, but if that'
                    ' specific version is missing (which seems to happen on Windows), it will'
                    ' raise an error.')

flags.DEFINE_string('vae_model', None, 'Path to the root directory of the Conv. VAE model.')
flags.DEFINE_bool('det_vae', True, 'Whether generated VAE obs are deterministic.')
flags.DEFINE_integer('max_eps', 0, 'Maximum number of episodes from which to collect data.')

flags.DEFINE_string('replays', None, 'Path to replay file(s).')
flags.DEFINE_string('output', 'output', 'Path to the directory in which to save the collected data.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')

flags.DEFINE_integer('video_eps', 0,
                     'Num. episodes for which to save videos of the screen, original and reconstructed SC2 frames.')
flags.DEFINE_float('fps', 22.5, 'The frames per second ratio used to save the videos.')
flags.DEFINE_integer('crf', 18, 'Video constant rate factor: the default quality setting in `[0, 51]`')

point_flag.DEFINE_point('window_size', '1024,768', 'SC2 window size.')
flags.DEFINE_bool('hide_hud', True, 'Whether to hide the HUD / information panel at the bottom of the screen.')

flags.mark_flags_as_required(['replays', 'experiment', 'env', 'vae_model'])


def process_replay(args, file, vae_model):
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

    # create real agent
    real_agent = SC2ReaverAgent(
        env.agent_interface_format, args.seed, args.results_dir, args.experiment, args.env,
        args.obs_features, args.action_set, args.action_spatial_dim,
        args.safety, args.safe_distance, args.safe_flee_angle_range, args.safe_margin, args.safe_unsafe_steps,
        args.gin_files, args.gin_bindings, args.agent, args.gpu)

    # create vae-based agent
    converter = ReaverVAESampleConverter(
        env.agent_interface_format, vae_model, args.action_set, args.action_spatial_dim)
    vae_agent = SC2ReaverVAEAgent(real_agent, converter)

    # collect datapoints for real and vae-reconstructed observations
    real_datapoints = real_agent.get_interaction_datapoints(agent_obs)
    vae_datapoints = vae_agent.get_interaction_datapoints((agent_obs, agent_actions))

    # collects real vs vae-reconstructed observation images and saves video
    real_obs_img_buffer = converter.to_images(agent_obs[:max_video_step])
    file_path = os.path.join(out_dir, '{}-orig-layers.mp4'.format(get_file_name_without_extension(file)))
    save_video(real_obs_img_buffer, file_path, args.fps, args.crf, verify_input=True, color='white')

    vae_obs_img_buffer = converter.to_images(vae_agent.agent_observations[:max_video_step])
    file_path = os.path.join(out_dir, '{}-vae-layers.mp4'.format(get_file_name_without_extension(file)))
    save_video(vae_obs_img_buffer, file_path, args.fps, args.crf, verify_input=True, color='white')

    # evaluates real vs VAE performance
    compare_datapoints(
        real_datapoints, vae_datapoints, out_dir, 'Real', 'VAE', [c.name for c in vae_model.components])

    # mean vae variance
    mean_vars = np.exp(vae_agent.latent_log_vars).mean(axis=0, keepdims=True).reshape(1, -1)
    plot_evolution(mean_vars, [''], 'Mean VAE Variance $z_{\sigma^2}$',
                   output_img=os.path.join(out_dir, 'eval-vae-var.pdf'), x_label='Time')

    logging.info('Finished after {} timesteps!'.format(len(screen_buffer)))


def main(unused_argv):
    args = flags.FLAGS

    # checks output dir and log file
    create_clear_dir(args.output, args.clear)
    change_log_handler(os.path.join(args.output, 'evaluate_vae.log'), args.verbosity)

    # save args
    with open(os.path.join(args.output, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    # loads VAE model
    vae_model = model_remapper.load_model(args.vae_model)

    # checks input files
    if os.path.isfile(args.replays):
        files = [args.replays]
    elif os.path.isdir(args.replays):
        files = list(get_files_with_extension(args.replays, 'SC2Replay'))
    else:
        raise ValueError('Input path is not a valid file or directory: {}.'.format(args.input))

    # process files
    for file in files:
        process_replay(args, file, vae_model)


if __name__ == '__main__':
    app.run(main)
