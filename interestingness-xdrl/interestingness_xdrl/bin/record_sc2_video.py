import json
import logging
import platform
import os
from absl import flags, app
from interestingness_xdrl.environments.sc2 import SC2Environment
from interestingness_xdrl.util.io import create_clear_dir, get_file_name_without_extension
from interestingness_xdrl.util.logging import change_log_handler
from interestingness_xdrl.util.video import save_video

"""
Loads replays and saves a video file of the main game screen.
"""

FLAGS = flags.FLAGS
flags.DEFINE_string('output', 'output', 'Path to the directory in which to save the video files.')
flags.DEFINE_bool('separate', False, 'Whether to separate videos by episodes.')
flags.DEFINE_float('fps', 22.5, 'The frames per second ratio used to save the videos.')
flags.DEFINE_integer('crf', 18, 'Video constant rate factor: the default quality setting in `[0, 51]`')
flags.DEFINE_bool('hide_hud', False, 'Whether to hide the HUD / information panel at the bottom of the screen.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')


def main(unused_args):
    args = flags.FLAGS

    # check for mac OS
    if platform.system() != 'Darwin':
        raise ValueError('Highlights extraction is currently not supported in non-macOS platforms.')

    # checks output dir and log file
    out_dir = args.output
    create_clear_dir(out_dir, args.clear)
    change_log_handler(os.path.join(out_dir, 'video.log'), args.verbosity)
    logging.info('===================================================================')

    # save args
    with open(os.path.join(out_dir, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    # collect images
    env = SC2Environment(
        args.replays, args.step_mul, args.replay_sc2_version, 1, args.window_size, args.hide_hud, True)
    env_data = env.collect_all_data()

    # saves single video or per episode # TODO more replays
    replay_name = get_file_name_without_extension(args.replays)
    frame_buffers = {}
    if args.separate:
        new_eps = list(env_data.new_episodes)
        new_eps.append(len(env_data.frames))
        frame_buffers.update({
            os.path.join(out_dir, '{}-{}.mp4'.format(replay_name, i)): env_data.frames[new_eps[i]:new_eps[i + 1]]
            for i in range(len(new_eps) - 1)})
    else:
        frame_buffers[os.path.join(out_dir, '{}.mp4'.format(replay_name))] = env_data.frames

    for video_file, frames in frame_buffers.items():
        logging.info('Got {} video frames, saving to {}...'.format(len(frames), video_file))
        save_video(frames, video_file, args.fps, args.crf)

    logging.info('Done!')


if __name__ == "__main__":
    app.run(main)
