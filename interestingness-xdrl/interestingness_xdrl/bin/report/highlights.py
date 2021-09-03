import json
import logging
import os
import platform
from absl import app, flags
from interestingness_xdrl.analysis.full import FullAnalysis
from interestingness_xdrl.environments.sc2 import SC2Environment
from interestingness_xdrl.reporting.highlights import HighlightsReport
from interestingness_xdrl.util.io import create_clear_dir, load_object, get_file_name_without_extension
from interestingness_xdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Extracts highlights, i.e., video summaries of an agent\'s interaction with the environment as ' \
                  'identified by the several introspection analyses and corresponding interaction data.'

flags.DEFINE_integer('step_mul', 8, 'Game steps per observation at which the data was collected.')
flags.DEFINE_string('replay_sc2_version', None,
                    'SC2 version to use for replay. Either "x.y.z" or "latest". If not specified,'
                    ' version is inferred from the replay file. This ought to work, but if that'
                    ' specific version is missing (which seems to happen on Windows), it will'
                    ' raise an error.')
flags.DEFINE_bool('hide_hud', True, 'Whether to hide the HUD / information panel at the bottom of the screen.')
flags.DEFINE_string('data', None,
                    'Pickle file containing the interaction data collected using `collect_data_*` scripts.')
flags.DEFINE_string('analysis', None,
                    'Pickle file containing the full introspection analysis collected using the `analyze` script.')
flags.DEFINE_integer('record_time_steps', 41, 'The number of environment time-steps to be recorded in each video.')
flags.DEFINE_integer('max_highlights_per_elem', 4,
                     'The maximum number of highlights to be recorded for each type of element.')
flags.DEFINE_float('fps', 30, 'The frames-per-second at which videos are to be recorded.')
flags.DEFINE_float('fade_ratio', 0.25, 'The ratio of frames to which apply a fade-in/out effect.')
flags.DEFINE_string('output', 'output/report-highlights', 'Path to the directory in which to save the highlights.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')

flags.mark_flags_as_required(['data', 'analysis'])

RECORD_STEP_MUL = 8  # 1 # step multiplication for frame recording (not data recording)


def main(unused_argv):
    args = flags.FLAGS

    # check for mac OS
    if platform.system() != 'Darwin':
        raise ValueError('Highlights extraction is currently not supported in non-macOS platforms.')

    # checks output dir and log file
    out_dir = os.path.join(args.output, get_file_name_without_extension(args.replays))
    create_clear_dir(out_dir, args.clear)
    change_log_handler(os.path.join(out_dir, 'highlights.log'), args.verbosity)
    logging.info('===================================================================')

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

    logging.info('___________________________________________________________________')
    logging.info('Collecting visual information from SC2 by replaying \'{}\'...'.format(args.replays))

    # collect images
    env = SC2Environment(
        args.replays, RECORD_STEP_MUL, args.replay_sc2_version, 1, args.window_size, args.hide_hud, True)
    env_data = env.collect_all_data()

    # collects and saves highlights
    explainer = HighlightsReport(analyses, out_dir, env_data.frames, args.step_mul / RECORD_STEP_MUL,
                                 args.record_time_steps, args.max_highlights_per_elem, args.fps, args.fade_ratio)
    explainer.create()
    logging.info('Finished after {} timesteps!'.format(len(env_data.frames)))


if __name__ == '__main__':
    app.run(main)
