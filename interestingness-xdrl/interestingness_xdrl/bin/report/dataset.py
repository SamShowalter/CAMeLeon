import json
import logging
import os
from absl import app, flags
from interestingness_xdrl.analysis.full import FullAnalysis
from interestingness_xdrl.reporting.dataset import DatasetReport
from interestingness_xdrl.util.io import create_clear_dir, load_object, get_file_name_without_extension
from interestingness_xdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Saves a dataset on a CSV file containing labels for the different interestingness dimensions ' \
                  'at each timestep of analysis.'

flags.DEFINE_string('data', None,
                    'Pickle file containing the interaction data collected using `collect_data_*` scripts.')
flags.DEFINE_string('analysis', None,
                    'Pickle file containing the full introspection analysis collected using the `analyze` script.')
flags.DEFINE_string('output', 'output/report-highlights', 'Path to the directory in which to save the highlights.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')

flags.mark_flags_as_required(['data', 'analysis'])


def main(unused_argv):
    args = flags.FLAGS

    # checks output dir and log file
    out_dir = os.path.join(args.output)
    create_clear_dir(out_dir, args.clear)
    change_log_handler(os.path.join(out_dir, 'dataset.log'), args.verbosity)
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

    # collects and saves datasets
    report = DatasetReport(analyses, out_dir)
    report.create()
    logging.info('Finished after {} timesteps!'.format(len(interaction_data)))


if __name__ == '__main__':
    app.run(main)
