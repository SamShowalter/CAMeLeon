import json
import logging
import os
import numpy as np
from absl import app, flags
from interestingness_xdrl.analysis.config import AnalysisConfiguration
from interestingness_xdrl.analysis.full import FullAnalysis
from interestingness_xdrl.util.io import create_clear_dir, load_object, get_directory_name
from interestingness_xdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Performs several analyses to extract interestingness elements according to previously-collected ' \
                  'interaction data.'

flags.DEFINE_string('data', None,
                    'Pickle file containing the interaction data collected using `collect_data_*` scripts.')
flags.DEFINE_string('config', 'config/analysis.json', 'Path to the analysis configuration file.')
flags.DEFINE_string('output', 'output/analyses', 'Path to the directory in which to save the analyses files.')
flags.DEFINE_string('img_format', 'pdf', 'Format of images to be saved during analysis.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')

flags.mark_flag_as_required('data')


def main(unused_argv):
    args = flags.FLAGS

    if not os.path.isfile(args.data):
        raise ValueError('Could not find interaction data file in {}'.format(args.data))

    # checks output dir and log file
    out_dir = os.path.join(args.output, get_directory_name(args.data))
    create_clear_dir(out_dir, args.clear)
    change_log_handler(os.path.join(out_dir, 'analyses.log'), args.verbosity)

    # save args
    with open(os.path.join(out_dir, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    # load data file
    interaction_data = load_object(args.data)
    num_eps = len(np.where([dp.new_episode for dp in interaction_data])[0])
    logging.info('Loaded interaction data corresponding to {} timesteps ({} episodes)from: {}'.format(
        len(interaction_data), num_eps, args.data))

    # load analysis config
    if not os.path.isfile(args.config):
        raise ValueError('Could not find analysis configuration file in {}'.format(args.config))
    config = AnalysisConfiguration.load_json(args.config)
    logging.info('Loaded analysis configuration file from: {}'.format(args.config))
    config.save_json(os.path.join(out_dir, os.path.basename(args.config)))

    # creates full analysis with all analyses
    analysis = FullAnalysis(interaction_data, config, args.img_format)
    logging.info('{} total analyses to be performed...'.format(len(analysis)))

    # runs and saves results
    analysis.analyze(out_dir)
    analysis.save(os.path.join(out_dir, 'analyses.pkl.gz'))


if __name__ == '__main__':
    app.run(main)
