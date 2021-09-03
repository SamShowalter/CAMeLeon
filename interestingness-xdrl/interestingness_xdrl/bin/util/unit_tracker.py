import json
import logging
import os
import cv2
import skvideo.io
import numpy as np
from PIL import Image
from absl import app, flags
from pysc2.lib import point_flag
from interestingness_xdrl.environments.sc2 import SC2Environment
from interestingness_xdrl.util.io import create_clear_dir
from interestingness_xdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

flags.DEFINE_integer('step_mul', 8, 'Game steps per observation at which the data was collected.')
flags.DEFINE_string('replay_sc2_version', 'latest',
                    'SC2 version to use for replay. Either "x.y.z" or "latest". If not specified,'
                    ' version is inferred from the replay file. This ought to work, but if that'
                    ' specific version is missing (which seems to happen on Windows), it will'
                    ' raise an error.')

flags.DEFINE_string('replay_file', None, 'Path to replay file.')
point_flag.DEFINE_point('window_size', '1024,768', 'SC2 window size.')

flags.DEFINE_float('fps', 22.5, 'The frames per second ratio used to save the videos.')
flags.DEFINE_integer('crf', 18, 'Video constant rate factor: the default quality setting in `[0, 51]`')
flags.DEFINE_bool('hide_hud', True, 'Whether to hide the HUD / information panel at the bottom of the screen.')
flags.DEFINE_string('output', 'output/tracker', 'Path to the directory in which to save the highlights.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')

flags.mark_flag_as_required('replay_file')

CAMERA_WIDTH = 24
FEATURE_DIMENSIONS = 256


def main(unused_argv):
    args = flags.FLAGS

    # checks output dir and log file
    create_clear_dir(args.output, args.clear)
    change_log_handler(os.path.join(args.output, 'tracker.log'), args.verbosity)
    logging.info('===================================================================')

    # save args
    with open(os.path.join(args.output, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    logging.info('___________________________________________________________________')
    logging.info('Tracking units in SC2 by replaying \'{}\'...'.format(args.replay_file))

    env = SC2Environment(
        args.replay_file, args.step_mul, 1., args.replay_sc2_version, 1, False, args.window_size, args.hide_hud, True,
        FEATURE_DIMENSIONS, CAMERA_WIDTH, True)
    env.start()

    # gets perspective transformation matrix
    left = int(0.04 * FEATURE_DIMENSIONS)
    right = int(0.96 * FEATURE_DIMENSIONS)
    top = int(0.04 * FEATURE_DIMENSIONS)
    bottom = int(0.7 * FEATURE_DIMENSIONS)
    m = cv2.getPerspectiveTransform(
        np.float32([(left, top), (right, top), (right, bottom), (left, bottom)]),
        np.float32([(0.09, 0.05), (0.91, 0.05), (1.04, 0.76), (-0.04, 0.76)]) * np.float32(env.visual_observation.size))

    # creates "square matrix"
    border_mat = np.zeros(tuple(env.agent_interface_format.feature_dimensions.screen))
    border_mat[0, :] = border_mat[-1, :] = border_mat[:, 0] = border_mat[:, -1] = 1

    ep = -1
    video_writer = None
    replay_file = os.path.basename(args.replay_file)
    while not env.finished:

        if env.t == 0 or env.new_episode:
            ep += 1

            if video_writer is not None:
                video_writer.close()

            # creates video writer
            ext_idx = replay_file.lower().find('.sc2replay')
            output_file = os.path.join(
                args.output, '{}-{}.mp4'.format(replay_file[:ext_idx], ep))
            video_writer = skvideo.io.FFmpegWriter(
                output_file, inputdict={'-r': str(args.fps)}, outputdict={'-crf': str(args.crf), '-pix_fmt': 'yuv420p'})

            logging.info('Recording episode {} of replay \'{}\' to \'{}\'...'.format(ep, replay_file, output_file))

        # capture units
        img = env.visual_observation
        if img is not None:
            masks_colors = [(border_mat, [255, 255, 255]),
                            (env.agent_obs.observation.feature_screen.player_relative == 1, [255, 0, 0]),
                            (env.agent_obs.observation.feature_screen.player_relative == 4, [0, 0, 255])]
            for mask, color in masks_colors:
                mask = np.asarray(mask * 100, dtype=np.uint8)
                mask = cv2.warpPerspective(mask, m, img.size)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
                mask = cv2.GaussianBlur(mask, (51, 51), 0)
                mask = Image.fromarray(mask)

                overlay = np.zeros((img.size[1], img.size[0], 3), dtype=np.uint8)
                for i in range(len(color)):
                    overlay[:, :, i] = color[i]
                overlay = Image.fromarray(overlay)

                img = Image.composite(overlay, img, mask)
            video_writer.writeFrame(np.array(img))

        env.step()

    env.stop()

    logging.info('Finished after {} timesteps!'.format(env.t))


if __name__ == '__main__':
    app.run(main)
