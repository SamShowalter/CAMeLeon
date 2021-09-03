import os
import logging
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pysc2.env.environment import TimeStep
from interestingness_xdrl.agents import Agent
from interestingness_xdrl.analysis.full import FullAnalysis
from interestingness_xdrl.reporting import CompetencyReport

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

OVERLAY_ALPHA = 100
OVERLAY_COLOR = [255, 0, 0]
OVERLAY_BLUR_SIZE = 51
OVERLAY_DILATE_SIZE = 10


class CounterfactualsReport(CompetencyReport):
    """
    An explainer that produces counterfactual examples for key moments of the agent's interaction with the environment.
    Counterfactuals are counter-examples for states that, if experienced by the agent, would lead to a very different
    situation with regards to a dimension of analysis from introspection.
    """

    def __init__(self, full_analysis, output_dir, visual_obs, agent):
        """
        Creates a new explainer.
        :param FullAnalysis full_analysis: the full introspection analysis over the agent's history of interaction
        with the environment.
        :param str output_dir: the path to the directory in which to save explanations / results.
        :param list[Image.Image] visual_obs: the visual representations of the environment at each step.
        :param Agent agent: the agent capable of creating counterfactuals.
        """
        super().__init__(full_analysis, output_dir)
        self._agent = agent
        self._visual_obs = visual_obs

    def create(self):

        # collect counterfactuals for all elements
        logging.info('===================================================================')
        logging.info('Collecting counterfactuals for {} timesteps...'.format(len(self.full_analysis.data)))
        analyses_counterfactuals = self._get_analyses_counterfactuals()
        logging.info('Collected counterfactuals for {} interestingness elements.'.format(len(analyses_counterfactuals)))

        # TODO
        # # check match with interaction data provided
        # assert len(self._visual_obs) == len(self.full_analysis.data), \
        #     'Num. interaction datapoints ({}) does not match environment replay steps ({})'.format(
        #         len(self.full_analysis.data), len(self._visual_obs))

        # gets difference and saves counterfactuals
        counterfactuals_info = {}
        for args in analyses_counterfactuals:
            full_name = args[0]
            if full_name not in counterfactuals_info:
                counterfactuals_info[full_name] = []
            counterfactuals_info[full_name].extend(self._save_counterfactuals(*args))

        # saves dataset with counterfactual info for each analysis dimension
        for full_name, infos in counterfactuals_info.items():
            header = ','.join(
                ['Timestep', 'Index', 'Element', 'Perceptual Distance', 'Value / Delta', 'Description'])
            file_path = os.path.join(full_name, 'counterfactuals-{}.csv'.format(os.path.basename(full_name)))
            np.savetxt(file_path, infos, '%s', ',', header=header, comments='')
            logging.info('Saved counterfactual information file for \'{}\' in: {}'.format(full_name, file_path))

        logging.info('===================================================================')

    def _get_analyses_counterfactuals(self):

        # iterate through all timesteps
        analyses_counterfactuals = []
        for t in tqdm(range(len(self._visual_obs)), 'Collecting counterfactuals'):
            datapoint = self.full_analysis.data[t]

            # collect the elements detected at this timestep, if any
            counterfactuals = []
            for level, dimension, analysis in self.full_analysis.elem_iterator:
                element, val = analysis.get_element_time(t)
                if element == '':
                    continue

                # only compute counterfactuals for this timestep if necessary
                if len(counterfactuals) == 0:
                    counterfactuals = self._agent.get_counterfactuals(datapoint)

                full_name = os.path.join(self.output_dir, level, dimension)
                analyses_counterfactuals.append(
                    (full_name, t, analysis, element, val, self._visual_obs[t], datapoint, counterfactuals))

        return analyses_counterfactuals

    def _save_counterfactuals(self, full_name, t, analysis, element, val, visual_obs, datapoint, counterfactuals):

        logging.info('___________________________________________________________________')
        logging.info('{}: {} counterfactuals for a \'{}\' element with value {:.3f} found at {}:'.format(
            full_name, len(counterfactuals), element, val, t))

        # saves original frame
        visual_obs.save(os.path.join(full_name, '{}-{}-orig.png'.format(element, t)))

        # generates counterfactual images and gathers info
        counterfactuals_info = [[t, -1, element, 0., val, '"original observation"']]
        for i in range(len(counterfactuals)):
            counterfactual, desc = counterfactuals[i]
            counter_element, counter_val = analysis.get_element_datapoint(counterfactual)
            dist = self._get_distance(counterfactual, datapoint)
            delta = counter_val - val

            logging.info('\t{}: element: {}, distance: {:.3f}, delta: {:.3f}'.format(
                i, counter_element, dist, delta))
            counterfactuals_info.append([t, i, counter_element, dist, delta, '"{}"'.format(desc)])

            # get and perceptual difference frame
            counter_obs = self._get_counterfactual_explanation(
                visual_obs, element, counter_element, datapoint, counterfactual)
            counter_obs.save(os.path.join(full_name, '{}-{}-counter-{}.png'.format(element, t, i)))

        return counterfactuals_info

    def _get_distance(self, datapoint1, datapoint2):
        if isinstance(datapoint1.observation, TimeStep):
            # return the Euclidean distance
            return np.linalg.norm(np.asarray(datapoint1.observation.observation.feature_screen) -
                                  np.asarray(datapoint2.observation.observation.feature_screen))

        raise NotImplementedError('Unable to process observation of type: {}'.format(type(datapoint1.observation)))

    def _get_counterfactual_explanation(self, visual_obs, elements, counter_elements, datapoint, counterfactual):
        if isinstance(datapoint.observation, TimeStep):

            # checks transformation matrix
            if not hasattr(self, 'm') or self.m is None:
                f_width = datapoint.observation.observation.feature_screen.unit_density.shape[0]
                f_height = datapoint.observation.observation.feature_screen.unit_density.shape[1]
                left = int(0.04 * f_width)
                right = int(0.96 * f_width)
                top = int(0.04 * f_height)
                bottom = int(0.7 * f_height)
                self.m = cv2.getPerspectiveTransform(
                    np.float32([(left, top), (right, top), (right, bottom), (left, bottom)]),
                    np.float32([(0.09, 0.05), (0.91, 0.05), (1.04, 0.76), (-0.04, 0.76)]) * np.float32(
                        visual_obs.size))

            # get the difference in the unit density layer as mask image
            mask = np.asarray(datapoint.observation.observation.feature_screen.unit_density -
                              counterfactual.observation.observation.feature_screen.unit_density, dtype=bool)
            mask = np.asarray(mask * OVERLAY_ALPHA, dtype=np.uint8)
            mask = cv2.warpPerspective(mask, self.m, visual_obs.size)
            mask = cv2.dilate(mask,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OVERLAY_DILATE_SIZE, OVERLAY_DILATE_SIZE)))
            mask = cv2.GaussianBlur(mask, (OVERLAY_BLUR_SIZE, OVERLAY_BLUR_SIZE), 0)
            mask = Image.fromarray(mask)

            # gets semi-transparent red overlay image
            overlay = np.zeros((visual_obs.size[1], visual_obs.size[0], 3), dtype=np.uint8)
            for i in range(len(OVERLAY_COLOR)):
                overlay[:, :, i] = OVERLAY_COLOR[i]
            overlay = Image.fromarray(overlay)

            # blends image
            return Image.composite(overlay, visual_obs, mask)

        raise NotImplementedError('Unable to process observation of type: {}'.format(type(datapoint.observation)))
