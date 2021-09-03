import json
import logging
import os
import shutil
import time
import numpy as np
import datetime
import multiprocessing as mp
from absl import app, flags
from sc2recorder.replayer import DebugReplayProcessor, DebugStepListener, ReplayProcessRunner
from interestingness_xdrl.util.io import create_clear_dir, get_files_with_extension, get_file_name_without_extension
from interestingness_xdrl.util.logging import change_log_handler
from interestingness_xdrl.util.plot import plot_evolution

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

REPLAY_EXT = 'SC2Replay'
PLAYER_PERSPECTIVE = 1
THRESH_STDS = 3

flags.DEFINE_string('output', None, 'Path to the directory in which to save the dynamics model')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results')
flags.DEFINE_bool('analyze', False, 'Whether to analyze the number of timesteps per replay episode')
flags.mark_flags_as_required(['replays', 'output'])


class _ReplayAnalyzerProcessor(DebugReplayProcessor):
    def __init__(self, steps_queue, step_mul=8):
        """
        Creates a new replay analyzer processor.
        :param mp.Queue steps_queue: the queue to report replay steps.
        :param int step_mul: the step multiplier for the SC2 environment.
        """
        self._step_mul = step_mul
        self._listener = _ReplayAnalyzerListener(steps_queue)

    @property
    def step_mul(self):
        return self._step_mul

    def create_listeners(self):
        return [self._listener]


class _ReplayAnalyzerListener(DebugStepListener):

    def __init__(self, steps_queue):
        """
        Creates a new replay analyzer listener.
        :param mp.Queue steps_queue: the queue to report replay steps and episodes.
        """
        self._steps_queue = steps_queue

        self._ignore_replay = False
        self._replay_name = ''
        self._ep_steps = 0
        self._ep_steps_list = []
        self._cur_ep = 0

    def start_replay(self, replay_name, replay_info, player_perspective):
        # ignore if not player's side
        self._ignore_replay = player_perspective != PLAYER_PERSPECTIVE

        if not self._ignore_replay:
            self._replay_name = replay_name
            self._ep_steps = 0
            self._ep_steps_list = []
            logging.info('Analyzing replay "{}"...'.format(replay_name))

    def finish_replay(self):
        if not self._ignore_replay:
            logging.info('Finished replay "{}", processed {} episodes, {} total steps'.format(
                self._replay_name, len(self._ep_steps_list), sum(self._ep_steps_list)))
            self._steps_queue.put(self._ep_steps_list)

    def step(self, ep, step, pb_obs, agent_obs, agent_actions):
        if self._ignore_replay:
            return

        if ep != self._cur_ep:
            logging.info('Episode {} ended in {} timesteps'.format(ep, self._ep_steps))
            self._ep_steps_list.append(self._ep_steps)
            self._cur_ep = ep
            self._ep_steps = 0

        self._ep_steps += 1


def main(unused_argv):
    args = flags.FLAGS

    # checks output dir and files
    create_clear_dir(args.output, args.clear)
    change_log_handler(os.path.join(args.output, 'select.log'), 1)

    # gets files' properties
    files = get_files_with_extension(args.replays, REPLAY_EXT)
    logging.info('=============================================')
    logging.info('Found {} replays in {}'.format(len(files), args.replays))
    files_attrs = []
    for file in files:
        date = os.path.getmtime(file)
        size = os.path.getsize(file)
        files_attrs.append((file, date, size))

    # sort and gets files' sizes
    files_attrs.sort(key=lambda f: f[1])
    sizes = np.array([file_attrs[2] for file_attrs in files_attrs])
    plot_evolution(sizes.reshape(1, -1), [''], 'Replay File Size',
                   output_img=os.path.join(args.output, 'files-size.pdf'), y_label='bytes')

    # selects and copies files
    size_diffs = sizes[1:] - sizes[:-1]
    size_diff_thresh = -(np.mean(np.abs(size_diffs)) + THRESH_STDS * np.std(np.abs(size_diffs)))
    max_size_idxs = np.where(size_diffs < size_diff_thresh)[0].tolist()
    max_size_idxs.append(len(sizes) - 1)  # add last replay idx
    logging.info('{} replay files selected:'.format(len(max_size_idxs)))
    total_steps = 0
    for i in range(len(max_size_idxs)):
        idx = max_size_idxs[i]
        file, date, size = files_attrs[idx]
        logging.info('Copying "{}" (Created: {:%m-%d %H:%M:%S}, Size: {}b) to "{}"'.format(
            file, datetime.datetime.fromtimestamp(date), size, args.output))
        out_file = os.path.join(args.output, f'eps_{i}.{REPLAY_EXT}')
        shutil.copy(file, out_file)

        # check also timesteps metadata file from CAML scenarios
        timesteps_file = os.path.join(os.path.dirname(file), '{}_timesteps.json'.format(
            get_file_name_without_extension(file)))
        if os.path.isfile(timesteps_file):
            # collect all timesteps info from the files
            json_steps = []
            for j in range(0 if i == 0 else max_size_idxs[i - 1], idx + 1):
                f = files_attrs[j][0]
                with open(os.path.join(os.path.dirname(f), '{}_timesteps.json'.format(
                        get_file_name_without_extension(f))), 'r') as fp:
                    json_steps.extend(json.load(fp))

            # save timesteps file with all steps info
            with open(os.path.join(args.output, f'eps_{i}_timesteps.json'), 'w') as fp:
                json.dump(json_steps, fp)
            total_steps += len(json_steps)
    if total_steps > 0:
        logging.info('{} total steps selected from replays.'.format(total_steps))

    # runs replays to get stats
    if args.analyze:
        steps_queue = mp.JoinableQueue()
        sample_processor = _ReplayAnalyzerProcessor(steps_queue, args.step_mul)
        replayer_runner = ReplayProcessRunner(args.output, sample_processor, args.replay_sc2_version,
                                              mp.cpu_count(), player_ids=PLAYER_PERSPECTIVE)
        replayer_runner.run()
        steps_queue.put(None)

        # process results
        steps = []
        while True:
            ep_steps = steps_queue.get()
            if ep_steps is None:
                break
            steps.extend(ep_steps)
            time.sleep(0.01)

        steps = np.array([steps])
        plot_evolution(steps, [''], 'Steps per Episode',
                       output_img=os.path.join(args.output, 'ep-steps.pdf'), y_label='Num. Steps')

        logging.info('=============================================')
        logging.info('Got {} episodes, {} total steps, mean: {}'.format(steps.shape[1], steps.sum(), steps.mean()))

    logging.info('Finished')


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == '__main__':
    app.run(main)
