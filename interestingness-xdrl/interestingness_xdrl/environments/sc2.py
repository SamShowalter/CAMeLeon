import logging
import os
import platform
import multiprocessing as mp
import threading as td
from tqdm import tqdm
from pysc2.env.environment import TimeStep, StepType
from pysc2.env.sc2_env import AgentInterfaceFormat, Dimensions, ActionSpace
from pysc2.lib.actions import FunctionCall
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2recorder.replayer import DebugReplayProcessor, DebugStepListener, ReplayProcessRunner
from interestingness_xdrl.environments import Environment, EnvironmentData
from interestingness_xdrl.util.mac_os import get_window_id, get_window_image

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

SC2_WINDOW_NAME = 'StarCraft II'
SC2_WINDOW_OWNER = 'SC2'


class SC2Environment(Environment):
    """
    Represents a StarCraft II environment that runs a given replay file.
    """

    def __init__(self, replays, step_mul=8, sc2_version='latest',
                 player_id=1, window_size=(640, 480), hide_hud=False, capture_screen=False,
                 feature_screen=64, feature_minimap=64, parallel=1, **aif_kwargs):
        """
        Creates a new SC2 environment replayer.
        :param str replays: the path to the replay file(s).
        :param int step_mul: the environment's step multiplier.
        :param str sc2_version: the replay file version.
        :param int player_id: the id of the player considered as the agent in the replay file.
        :param (int, int) window_size: SC2 window size.
        :param bool hide_hud: whether to hide the HUD / information panel at the bottom of the screen.
        :param bool capture_screen: whether to capture images of the SC2 screen window.
        :param int or (int, int) feature_screen: the dimensions of the feature layers.
        :param int or (int, int) feature_minimap: the dimensions of the feature layers.
        :param bool use_feature_units: whether to capture feature units vector.
        :param bool crop_to_playable_area: whether to crop to playable area.
        :param int parallel: number of parallel processes used to collect data.
        :param aif_kwargs: extra arguments to build agent interface format.
        """
        super().__init__(0)  # not used here since we take all info from the replay

        self._replays = os.path.abspath(replays)
        self._step_mul = step_mul
        self._sc2_version = sc2_version
        self._player_id = player_id
        self._parallel = parallel
        self._capture_screen = capture_screen and platform.system() == 'Darwin'  # currently only works in mac OS

        if 'action_space' not in aif_kwargs:
            aif_kwargs.update({'action_space': ActionSpace.FEATURES})
        self._aif = AgentInterfaceFormat(
            feature_dimensions=Dimensions(feature_screen, feature_minimap),
            rgb_dimensions=Dimensions(window_size[0], 1) if hide_hud else None,
            **aif_kwargs)

    @property
    def agent_interface_format(self):
        return self._aif

    def collect_all_data(self, max_eps=0):
        """
        Collects all data from this environment by running the replay for a specified amount of episodes.
        :param int max_eps: the maximum number of episodes to be run. `<=0` will run all episodes.
        :rtype: EnvironmentData
        :return: the data collected during the agent's interaction with the environment.
        """

        # creates queues
        sc2_producer_queue = mp.JoinableQueue()
        consumer_queue = mp.JoinableQueue()

        # creates and starts consumer thread
        consumer_thread = _DataCollectionThread(sc2_producer_queue, consumer_queue, self._capture_screen)
        consumer_thread.start()

        # creates and runs the replay processor
        sample_processor = _ReplayCollectorProcessor(self._aif, sc2_producer_queue, self._step_mul, self._player_id)
        replayer_processor = ReplayProcessRunner(
            self._replays, sample_processor, self._sc2_version, self._parallel, player_ids=self._player_id)
        replayer_processor.run()
        sc2_producer_queue.put(None)  # no more samples

        # aggregates all data in the queue
        frames = []
        observations = []
        actions = []
        new_episodes = []
        steps = consumer_queue.get()
        consumer_queue.task_done()
        for t in tqdm(range(steps), 'Receiving SC2 data'):
            env_data = consumer_queue.get()
            consumer_queue.task_done()
            if env_data is None:
                break

            screen_frame, agent_obs, agent_actions, new_eps, total_steps = env_data
            if total_steps != t:
                logging.info('Incorrect order found in data, got t={}, expected {}'.format(total_steps, t))
            t += 1

            if new_eps:
                new_episodes.append(total_steps)
            observations.append(agent_obs)
            actions.append(agent_actions)
            frames.append(screen_frame)

        consumer_queue.get()
        consumer_queue.task_done()
        consumer_thread.join()
        logging.info('Finished collecting {} samples ({} episodes).'.format(len(observations), len(new_episodes)))
        return EnvironmentData(frames, observations, actions, new_episodes)


class _ReplayCollectorProcessor(DebugReplayProcessor):
    def __init__(self, aif, samples_queue, step_mul=8, player_id=1):
        """
        Creates a new SC2 environment replayer.
        :param AgentInterfaceFormat aif: the SC2 agent interface options.
        :param mp.Queue samples_queue: the queue to put data in.
        :param int step_mul: the environment's step multiplier.
        :param int player_id: the id of the player considered as the agent in the replay file.
        """
        self._aif = aif

        self._interface_options = sc_pb.InterfaceOptions(
            raw=True, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=self._aif.camera_width_world_units))
        self._aif.feature_dimensions.screen.assign_to(self._interface_options.feature_layer.resolution)
        self._aif.feature_dimensions.minimap.assign_to(self._interface_options.feature_layer.minimap_resolution)
        if self._aif.rgb_dimensions is not None:
            self._aif.rgb_dimensions.screen.assign_to(self._interface_options.render.resolution)
            self._aif.rgb_dimensions.minimap.assign_to(self._interface_options.render.minimap_resolution)

        self._listener = _ReplayCollectorListener(samples_queue, player_id)
        self._step_mul = step_mul

    @property
    def step_mul(self):
        return self._step_mul

    @property
    def interface(self):
        return self._interface_options

    @property
    def agent_interface_format(self):
        return self._aif

    def create_listeners(self):
        return [self._listener]


class _ReplayCollectorListener(DebugStepListener):

    def __init__(self, samples_queue, player_id):
        """
        Creates a new replay sampler listener.
        :param mp.Queue samples_queue: the queue to put the SC2 data in.
        :param int player_id: the id of the player considered as the agent in the replay file.
        """
        self._samples_queue = samples_queue
        self._player_id = player_id

        self._ignore_replay = False
        self._replay_name = ''
        self._total_eps = 0
        self._total_steps = 0

    def start_replay(self, replay_name, replay_info, player_perspective):
        """
        Called when starting a new replay.
        :param str replay_name: replay file name.
        :param ResponseReplayInfo replay_info: protobuf message.
        :param int player_perspective: ID of player whose perspective we see observations.
        :return:
        """
        # ignore if not player's side
        self._ignore_replay = player_perspective != self._player_id

        if not self._ignore_replay:
            self._replay_name = replay_name
            logging.info('[ReplayCollector] Collecting data from replay \'{}\'...'.format(replay_name))

            # send msg to thread, wait for ack
            self._samples_queue.put('start')
            self._samples_queue.join()

    def finish_replay(self):
        """
        Finished a replay.
        """
        if not self._ignore_replay:
            logging.info('Collected {} samples from {} episodes from replay \'{}\'...'.format(
                self._total_steps, self._total_eps, self._replay_name))

    def reset(self, pb_obs, agent_obs):
        pass

    def step(self, ep, step, pb_obs, agent_obs, agent_actions):
        """
        Puts the given observations in the queue.
        :param int ep: the episode that this observation was made.
        :param int step: the episode time-step in which this observation was made.
        :param ResponseObservation pb_obs: the observation in protobuf form.
        :param TimeStep agent_obs: the observation in pysc2 features form.
        :param list[FunctionCall] agent_actions: list of actions executed by the agent between the previous observation
        and the current observation.
        :return:
        """
        if self._ignore_replay:
            return

        # checks new episode
        if step == 0:
            self._total_eps += 1
            # force step type in observation
            agent_obs = TimeStep(StepType.FIRST, agent_obs.reward, agent_obs.discount, agent_obs.observation)

        # put sample in queue and wait for ack
        self._samples_queue.put((agent_obs, agent_actions, step == 0, self._total_steps))
        self._samples_queue.join()
        self._total_steps += 1


class _DataCollectionThread(td.Thread):

    def __init__(self, in_data_queue, out_data_queue, capture_screen=False, idx=0):
        """
        Creates a new thread to receive SC2 data and capture window frames.
        :param mp.Queue in_data_queue: the queue from which to fetch the data from the SC2 producer process.
        :param mp.Queue out_data_queue: the queue on which to place the data to be retrieved by the main process.
        :param bool capture_screen: whether to capture images of the SC2 screen window.
        :param int idx: the index of the SC2 producer process, used to identify  the SC2 screen window if multiple are
        active.
        """
        super().__init__()
        self._in_queue = in_data_queue
        self._out_queue = out_data_queue
        self._capture_screen = capture_screen
        self._idx = idx
        self._sc2_window_id = -1
        self._data = []

    def run(self):

        # wait for data
        while True:

            sc2_data = self._in_queue.get()
            if sc2_data is None:
                # stop was sent, closing
                logging.info('[DataCollector] Received None, sending {} samples...'.format(len(self._data)))
                self._out_queue.put(len(self._data))  # first send total samples
                self._out_queue.join()
                for data in self._data:
                    self._out_queue.put(data)
                    self._out_queue.join()
                self._out_queue.put(None)  # signal end
                self._out_queue.join()
                logging.info('[DataCollector] Done, exiting.')
                self._data = []
                return

            if sc2_data == 'start':
                if self._capture_screen:
                    # reset and try to get window
                    self._sc2_window_id = -1
                    ids = list(get_window_id(SC2_WINDOW_NAME, SC2_WINDOW_OWNER))
                    if len(ids) <= self._idx:
                        logging.info(
                            '[DataCollector] Could not find StarCraftII window for idx {} ({} available)!'.format(
                                self._idx, len(ids)))
                    else:
                        self._sc2_window_id = ids[self._idx]
                        logging.info('[DataCollector] Found SC2 window for idx {}: {}'.format(
                            self._idx, self._sc2_window_id))

            else:
                # get and check data from sc2 replay data producer
                agent_obs, agent_actions, new_eps, total_steps = sc2_data
                if total_steps != len(self._data):
                    logging.info('[DataCollector] Invalid data received: timestep {} when expecting {}'.format(
                        total_steps, len(self._data)))

                screen_frame = None
                if self._capture_screen and self._sc2_window_id != -1:
                    # take screenshot and append to buffer
                    screen_frame = get_window_image(self._sc2_window_id)

                # put data in output queue
                self._data.append((screen_frame, agent_obs, agent_actions, new_eps, total_steps))

            # send a confirmation to resume processing on the other side
            self._in_queue.task_done()
