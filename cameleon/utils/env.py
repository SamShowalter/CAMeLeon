
#################################################################################
#
#             Project Title:  General utilities for Cameleon Env
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import ast
from operator import add
import importlib

#Import information from Gym minigrid
from gym_minigrid.wrappers import ImgObsWrapper
from gym_minigrid.rendering import *

# Ray registry for RLlib
from ray.rllib.agents.registry import _get_trainer_class
from ray.tune.logger import Logger, UnifiedLogger

# Import information from cameleon
from cameleon.base_objects import *
from cameleon.envs import *
from cameleon.wrappers import *

#################################################################################
#   Function-Class Declaration
#################################################################################

def _render_tile(obj,
                tile_size=32,
                subdivs=3):
    """
    Render a tile and cache the result

    :obj: WorldObj: Object to place
    :highlight: bool: Whether or not to highlight
    :tile_size: Int: Render tile size
    :subdivs: Int: Subdivisions
    """

    # Hash map lookup key for the cache
    # Need to look into this hash map
    img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

    # Draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    if obj != None:
        obj.render(img)

    # Downsample the image to perform supersampling/anti-aliasing
    img = downsample(img, subdivs)

    return img

def render_encoded_env(
    encoded_env,
    tile_size=32,
    subdivs = 3):
    """
    Render this grid at a given scale

    :tile_size: Int:             Render tile size
    :highlight_mask: np.ndarray: Highlight mask

    """

    # Compute the total grid size
    width, height = encoded_env.shape[:2]
    width_px = width * tile_size
    height_px = height * tile_size

    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    # Render the grid
    for j in range(0, height):
        for i in range(0, width):
            obj = IDX_TO_INIT_OBJECT[encoded_env[i,j,0]]
            if obj:
                obj.color = IDX_TO_COLOR[encoded_env[i,j,1]]

            tile_img = _render_tile(
                obj,
                tile_size=tile_size,
                subdivs = subdivs
            )

            ymin = j * tile_size
            ymax = (j+1) * tile_size
            xmin = i * tile_size
            xmax = (i+1) * tile_size
            img[ymin:ymax, xmin:xmax, :] = tile_img

    return img



def cameleon_logger_creator(custom_path):
    """
    Customize way that cameleon saves off information

    :custom_path: Custom logging path

    """
    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = custom_path
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator

def update_config(config,config_update):
    """Update config with new keys. This only
    does key checking at a single layer of depth,
    but can accommodate dictionary assignment

    :config: Configuration
    :config_update: Updates to configuration
    :returns: config

    """
    for key, value in config_update.items():
        # assert key in config, "ERROR: Unknown key {} passed from config update to config."\
        #     .format(key)

        config[key] = value

    return config

def str2framework(s):
    """Make sure RLlib uses a compatible framework.
    RLlib natively supports tf, tf2, and torch, but
    using lazy evaluation with tf leads to issues.
    As of now, options must be torch or tf2

    :s: Framework string

    """
    if (not s):
        return None
    s = s.lower()
    assert s in ["tf","tf2","torch"],\
        "ERROR: framework {} not supported: Please used tf, tf2, or torch"\
                                .format(s)
    return s

def str2list(s):
    """Convert string to list,
    just gives peace of mind on CLI

    :s: Str: comma delimited string

    """
    if (not s) or (s == ""):
        return None
    slist = s.split(",")
    assert len(slist) > 0,\
    "ERRROR: str2list could not effectively parse string - {}".format(s)

    return slist

def str2str(s):
    """Test if a string exists

    :s: String

    """
    if (s == ""):
        return None
    return s


def str2int(s):
    """Test if a string exists

    :s: String

    """
    if (s == ""):
        return None
    assert int(s), "ERROR: input is not an integer"
    return int(s)

def wrap_env(env,wrappers):
    """Wrap environment

    :env: TODO
    :returns: TODO

    """
    for i in range(len(wrappers)):
        if isinstance(wrappers[i], tuple):
            env = wrappers[i][0](env,
                       agent_view_size = wrappers[i][1])
        else:
            env = wrappers[i](env)

    return env

def load_env(env_id):

    # Get environment spec
    env_spec = gym.envs.registry.env_specs[env_id]

    #Get entry point
    env_spec_name = env_spec.entry_point

    # Load the environment
    mod_name, attr_name = env_spec_name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)

    return fn()

def str2wrapper(wrappers):
    """Converts string into wrapper for later environment use
    Commas are used to separate wrappers

    :wrappers: List of wrapper strings, may be empty
    :returns: List of wrappers

    """
    if wrappers == "":
        return []
    wrappers = wrappers.split(",")
    wrapper_zoo = {
        "partial_obs": PartialObsWrapper,
        "encoding_only": ImgObsWrapper,
        "rgb_only": RGBImgObsWrapper,
        "canniballs_one_hot":CanniballsOneHotWrapper,
    }

    final_wrappers = []
    for w in wrappers:
        w_tags = w.split(".")

        # Extra information, like partial obs size
        if (len(w_tags) > 1):
            partial_obs_size = int(w_tags[1])
            wrapper = wrapper_zoo[w_tags[0]]
            final_wrappers.append((wrapper, partial_obs_size))

        else:
            final_wrappers.append(wrapper_zoo[w_tags[0]])

    return final_wrappers


def str2model(model_string, config = True):
    ms = model_string.upper()
    model,config = _get_trainer_class(ms, return_config = config)
    return model, config.copy()


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False, "Error: Input to str2bool unexpected - {}".format(v)

def str2dict(d_s):
    """Convert string to dictionary

    :d_s: Dictionary string
    :returns: Evaluated dictionary

    """
    return ast.literal_eval(d_s)

def dict2str(d):
    return str(d)

#######################################################################
# Main
#######################################################################

