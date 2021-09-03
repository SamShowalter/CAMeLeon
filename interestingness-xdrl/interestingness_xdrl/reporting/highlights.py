import os
import logging
import itertools
import numpy as np
from PIL import Image
from interestingness_xdrl.analysis.full import FullAnalysis
from interestingness_xdrl.reporting import CompetencyReport
from interestingness_xdrl.util.video import fade_video, save_video

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class HighlightsReport(CompetencyReport):
    """
    An explainer that captures short video clips highlighting important moments of the agent's interaction with its
    environment as dictated by the several extracted interestingness elements. The goal is to summarize the agent's
    aptitude in the task, both in terms of its capabilities and limitations according to different criteria.
    See [1]; inspired by the approach in [2].
    [1] - Sequeira, P., & Gervasio, M. (2019). Interestingness Elements for Explainable Reinforcement Learning:
    Understanding Agents' Capabilities and Limitations. arXiv preprint arXiv:1912.09007.
    [2] - Amir, D., & Amir, O. (2018). Highlights: Summarizing agent behavior to people. In AAMAS 2018.
    """

    def __init__(self, full_analysis, output_dir, frame_buffer, step_mul_ratio,
                 record_time_steps, max_highlights_per_elem, fps, fade_ratio):
        """
        Creates a new highlights extractor.
        :param FullAnalysis full_analysis: the full introspection analysis over the agent's history of interaction
        with the environment. Has to contain valid (non-None) visual observations, otherwise an error will be raised.
        :param str output_dir: the path to the directory in which to save explanations / results.
        :param list[Image.Image] frame_buffer: a list of images containing the visual observations used to
        produce the video highlights. Has to have the same length as the collected and analyzed data.
        :param float step_mul_ratio: the ratio denoting the number of frames in `frame_buffer` corresponding to one
        timestep of interestingness analysis.
        :param int record_time_steps: the number of environment time-steps to be recorded in each video.
        :param int max_highlights_per_elem: the maximum number of highlights to be recorded for each type of element.
        :param float fps: the frames-per-second at which videos are to be recorded.
        :param float fade_ratio: the ratio of frames to which apply a fade-in/out effect.
        """
        super().__init__(full_analysis, output_dir)
        assert full_analysis.data is not None and len(frame_buffer) > 0 and frame_buffer[0] is not None, \
            'Interaction data must include recorded visual observations to generate the visual summaries.'

        self._frame_buffer = frame_buffer
        self._step_mul_ratio = step_mul_ratio
        self._record_time_steps = record_time_steps
        self._max_highlights_per_elem = max_highlights_per_elem
        self._fps = fps
        self._fade_ratio = fade_ratio

        # shortcuts
        self._timer_timesteps = int((record_time_steps - 1) / 2)
        self._values = np.array([datapoint.value for datapoint in self.full_analysis.data])
        self._timesteps = len(self.full_analysis.data)

        self._candidate_highlights = {}

    def create(self):
        logging.info('===================================================================')
        logging.info('Extracting highlights from {} interaction timesteps...'.format(len(self.full_analysis.data)))

        # iterates through all the timesteps in the interaction data,
        # collecting timesteps in which elements were detected
        for t in range(self._timer_timesteps, self._timesteps - self._timer_timesteps):
            for level, dimension, analysis in self.full_analysis.elem_iterator:
                # gets element at this time-step and add to list
                elem, _ = analysis.get_element_time(t)
                if elem != '':
                    full_name = os.path.join(self.output_dir, level, dimension, elem)
                    if full_name not in self._candidate_highlights:
                        self._candidate_highlights[full_name] = []
                    self._candidate_highlights[full_name].append(t)

        # creates and saves video highlights
        for full_name, highlights in self._candidate_highlights.items():
            dimension = os.path.basename(os.path.dirname(full_name))
            level = os.path.basename(os.path.dirname(os.path.dirname(full_name)))
            logging.info('___________________________________________________________________')
            logging.info('Selecting highlights for dimension \'{}\' of introspection level \'{}\'...'.format(
                dimension, level))
            self._save_highlights(full_name, self._select_highlights(full_name))

    def _select_highlights(self, full_name):
        """
        Selects highlights for the given element by cutting-off according to diversity.
        :param str full_name: the full internal name identifier for the element.
        :rtype: list[int]
        :return: the list of time indexes corresponding to when the highlights should be captured.
        """

        # checks num highlights, return original list of does not surpass max
        highlights = np.array(self._candidate_highlights[full_name])
        if len(highlights) <= self._max_highlights_per_elem:
            return highlights

        # gets distances between all obs
        num_elems = len(highlights)
        distances = np.zeros((num_elems, num_elems))
        value_range = np.max(self._values) - np.min(self._values)
        for i in range(num_elems):
            for j in range(i + 1, num_elems):
                # todo other diffs ?
                value_diff = abs(self._values[highlights[i]] - self._values[highlights[j]]) / value_range
                time_diff = abs(highlights[i] - highlights[j]) / self._timesteps
                distances[i, j] = distances[j, i] = np.linalg.norm([value_diff, time_diff])

        # gets all combinations between highlights, selects the one maximizing inter-element diversity
        max_div = np.float('-inf')
        max_div_comb = []
        for elem_idx_comb in itertools.combinations(np.arange(num_elems), self._max_highlights_per_elem):
            # gets all pairwise dissimilarities
            pairs_dists = np.array([distances[pair] for pair in itertools.combinations(elem_idx_comb, 2)])

            # maximize both the maximum and the minimum distances
            diversity = np.min(pairs_dists) * np.max(pairs_dists)
            if diversity > max_div:
                max_div = diversity
                max_div_comb = highlights[list(elem_idx_comb)].tolist()
        return max_div_comb

    def _save_highlights(self, full_name, highlights):
        highlight_info = []
        for i, t in enumerate(highlights):
            # add fade-in/out effects to frame sequence
            start_t = int((t - self._timer_timesteps) * self._step_mul_ratio)
            end_t = int((t + self._timer_timesteps) * self._step_mul_ratio)
            buffer = fade_video(self._frame_buffer[start_t:end_t + 1], self._fade_ratio)
            if len(buffer) == 0:
                logging.info('Insufficient frames for highlight {}-{}'.format(full_name, t))
                continue

            # save highlight video to file
            video_file_path = '{}-{}.mp4'.format(full_name, t)
            save_video(buffer, video_file_path, self._fps)
            logging.info('Saved video highlight to {}'.format(video_file_path))

            # save screenshot
            self._frame_buffer[int(t * self._step_mul_ratio)].save(os.path.join('{}-{}.png'.format(full_name, t)))

            # add metadata
            highlight_info.append((os.path.basename(full_name), i, t, start_t, end_t))

        # save highlight frames to csv file in format (element, idx, frame_start, frame_end)
        file_path = '{}-highlights.csv'.format(full_name)
        np.savetxt(file_path, highlight_info, '%s', ',', comments='', header='element, idx, t, frame_start, frame_end')
        logging.info('Saved highlights info to {}'.format(file_path))
