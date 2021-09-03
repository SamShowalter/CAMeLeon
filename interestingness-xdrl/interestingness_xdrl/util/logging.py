import logging

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def change_log_handler(log_file, verbosity=1, fmt='[%(asctime)s %(levelname)s] %(message)s'):
    """
    Changes logger to use the given file.
    :param str log_file: the path to the intended log file.
    :param int verbosity: the level of verbosity, the higher the more severe messages will be logged.
    :param str fmt: the formatting string for the messages.
    :return:
    """
    log = logging.getLogger()
    for handler in log.handlers[:]:
        log.removeHandler(handler)
    file_handler = logging.FileHandler(log_file, 'w')
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    log.level = logging.WARN if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
