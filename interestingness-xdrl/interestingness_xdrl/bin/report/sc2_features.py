import json
import logging
import os
from absl import app, flags
from interestingness_xdrl.analysis.full import FullAnalysis
from interestingness_xdrl.reporting.sc2_features import SC2FeatureReport
from interestingness_xdrl.util.io import create_clear_dir
from interestingness_xdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Extracts high-level features to help describe moments of an agent\'s interaction with the ' \
                  'environment identified by the several introspection analyses and corresponding interaction data.'

flags.DEFINE_string('features', None,
                    'CSV file containing the features extracted at each timestep, e.g., extracted using the '
                    '`feature_extractor` module.')
flags.DEFINE_string('analysis', None,
                    'Pickle file containing the full introspection analysis collected using the `analyze` script.')
flags.DEFINE_integer('time_steps', 11, 'The number of environment time-steps prior to each element to be analyzed.')
flags.DEFINE_string('output', 'output/highlights', 'Path to the directory in which to save the extracted features.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')


def main(unused_argv):
    args = flags.FLAGS

    # checks output dir and log file
    create_clear_dir(args.output, args.clear)
    change_log_handler(os.path.join(args.output, 'features.log'), args.verbosity)
    logging.info('===================================================================')

    # save args
    with open(os.path.join(args.output, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    # load full analysis
    if not os.path.isfile(args.analysis):
        raise ValueError('Could not find full analysis data file in {}'.format(args.analysis))
    analyses = FullAnalysis.load(args.analysis)
    logging.info('Loaded full analysis data file from: {}'.format(args.analysis))

    # collects and saves features
    explainer = SC2FeatureReport(analyses, args.output, args.features, args.time_steps)
    explainer.create()


if __name__ == '__main__':
    flags.mark_flag_as_required('analysis')
    app.run(main)
