import gzip
import os
import pickle
import shutil
from tqdm import tqdm
import math
from multiprocessing.pool import ThreadPool
import re
import pathlib
from glob import glob


__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def get_file_changed_extension(file, ext):
    """
    Changes the extension of the given file.
    :param str file: the path to the file.
    :param str ext: the new file extension.
    :rtype: str
    :return: the file path with the new extension.
    """
    return os.path.join(os.path.dirname(file),
                        '{}.{}'.format(get_file_name_without_extension(file), ext.replace('.', '')))


def get_file_name_without_extension(file):
    """
    Gets the file name in the given path without extension.
    :param str file: the path to the file.
    :rtype: str
    :return: the file name in the given path without extension.
    """
    file = os.path.basename(file)
    return file.replace(os.path.splitext(file)[-1], '')


def get_files_with_extension(path, extension, sort=True):
    """
    Gets all files in the given directory with a given extension.
    :param str path: the directory from which to retrieve the files.
    :param str extension: the extension of the files to be retrieved.
    :param bool sort: whether to sort list of files based on file name.
    :rtype: list[str]
    :return: the list of files in the given directory with the required extension.
    """
    file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.' + extension)]
    if sort:
        file_list.sort()
    return file_list


def get_directory_name(path):
    """
    Gets the directory name in the given path.
    :param str path: the path (can be a file).
    :rtype: str
    :return: the directory name in the given path.
    """
    return os.path.basename(os.path.dirname(path))


def create_clear_dir(path, clear=False):
    """
    Creates a directory in the given path. If it exists, optionally clears the directory.
    :param str path: the path to the directory to create/clear.
    :param bool clear: whether to clear the directory if it exists.
    :return:
    """
    if clear and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def save_object(obj, file_path, compress_gzip=True):
    """
    Saves a binary file containing the given data.
    :param obj: the object to be saved.
    :param str file_path: the path of the file in which to save the data.
    :param bool compress_gzip: whether to gzip the output file.
    :return:
    """
    with gzip.open(file_path, 'wb') if compress_gzip else open(file_path, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(file_path):
    """
    Loads an object from the given file, possibly gzip compressed.
    :param str file_path: the path to the file containing the data to be loaded.
    :return: the data loaded from the file.
    """
    try:
        with gzip.open(file_path, 'rb') as file:
            return pickle.load(file)
    except OSError:
        with open(file_path, 'rb') as file:
            return pickle.load(file)


def _save_episode(input_tuple):
    """ Convenience routine for unboxing singleton input datums required by pooling"""
    episode, episode_fpath = input_tuple
    save_object(episode, episode_fpath)
        
        
def save_episodes(episodes, episodes_root, num_workers=5):
    """ Given a list of episodes, each a list of observation points, saves the 
    episodes to disk individually to facilitate faster loading.  This saves
    everything in conjunction using a ThreadPool, and set num_workers to adjust 
    number of concurrent runners.
    """
    num_zeros = 1 + math.ceil(math.pow(len(episodes),  1/10)) # Count number of zeros to pad with
    episode_fmtstr = "episode_{:0"+str(num_zeros)+"d}.pkl.gz"
    # Construct the schedule
    eps_schedule = [(episodes[i], os.path.join(episodes_root, episode_fmtstr.format(i)))
                     for i in range(len(episodes))]
    
    os.makedirs(episodes_root, exist_ok=True)
    with ThreadPool(num_workers) as p:
        list(tqdm(p.imap_unordered(_save_episode, eps_schedule), total = len(episodes)))
        

def _load_episode(episode_fpath):
    """ Convenience routine for load_episodes """
    episode = load_object(episode_fpath)
    fname = pathlib.Path(episode_fpath).name
    episode_number = 0
    m = re.search("episode_(\\d+).pkl*", fname)
    if m is not None:
        episode_number = int(m.group(1))
    return (episode_number, episode)


def load_episodes(episodes_root, num_workers=5):
    """ Given the root directory where episodes were saved out using save_episodes, loads 
    episodes into memory in order.  Uses parallelized reads via multiprocessing."""
    read_episodes = []
    episode_fpaths = glob(os.path.join(episodes_root, "*.pkl.gz"))
    with ThreadPool(num_workers) as p:
        read_episodes = list(tqdm(p.imap_unordered(_load_episode, episode_fpaths), total = len(episode_fpaths)))
    read_episodes.sort(key=lambda x: x[0])
    return [x[1] for x in read_episodes]
