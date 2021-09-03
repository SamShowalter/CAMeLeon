import platform
from PIL import Image

try:
    import Quartz.CoreGraphics as CG
except ImportError:
    pass  # won't be able to capture window screen

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

CG_WINDOW_NUMBER = 'kCGWindowNumber'
CG_WINDOW_OWNER_NAME = 'kCGWindowOwnerName'
CG_WINDOW_NAME = 'kCGWindowName'
CG_WINDOW_ON_SCREEN = 'kCGWindowIsOnscreen'
TITLE_BAR_HEIGHT = 56


def get_window_id(name, owner=None, on_screen=True):
    """
    Gets the id of the window with the given name. Only valid in macOS.
    :param str name: the name of the window whose id we want to retrieve.
    :param str owner: the name of the window owner whose id we want to retrieve.
    :param bool on_screen: require the window to have the "on screen" flag with a `True` value.
    :rtype: list[int]
    :return: the id of the window or -1 if no window with the given name was found.
    """
    # check for mac OS
    if platform.system() != 'Darwin':
        raise ValueError('Not supported in non-macOS platforms!')

    # get list of windows, search for name or owner and state and return id
    wl = CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionAll, CG.kCGNullWindowID)
    for window in wl:
        if CG_WINDOW_NAME in window and name == window[CG_WINDOW_NAME] and \
                (not on_screen or (CG_WINDOW_ON_SCREEN in window and window[CG_WINDOW_ON_SCREEN])):
            yield int(window[CG_WINDOW_NUMBER])
        elif owner is not None and CG_WINDOW_OWNER_NAME in window and owner == window[CG_WINDOW_OWNER_NAME] and \
                (not on_screen or (CG_WINDOW_ON_SCREEN in window and window[CG_WINDOW_ON_SCREEN])):
            yield int(window[CG_WINDOW_NUMBER])


def get_window_image(window_d, crop_title=True):
    """
    Gets an image object of the contents of the given window. Only valid in macOS.
    See: https://stackoverflow.com/a/53607100
    See: https://stackoverflow.com/a/22967912
    :param int window_d: the id of the window that we want to capture.
    :param bool crop_title: whether to crop the title bar part of the window.
    :rtype: PIL.Image.Image
    :return: the image representation of the given window.
    """
    # check for mac OS
    if platform.system() != 'Darwin':
        raise ValueError('Not supported in non-macOS platforms!')

    # get CG image
    cg_img = CG.CGWindowListCreateImage(
        CG.CGRectNull,
        CG.kCGWindowListOptionIncludingWindow,
        window_d,
        CG.kCGWindowImageBoundsIgnoreFraming | CG.kCGWindowImageBestResolution)

    width = CG.CGImageGetWidth(cg_img)
    height = CG.CGImageGetHeight(cg_img)
    pixel_data = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(cg_img))
    bpr = CG.CGImageGetBytesPerRow(cg_img)

    # create image and crop title
    img = Image.frombuffer('RGBA', (width, height), pixel_data, 'raw', 'BGRA', bpr, 1)
    if crop_title:
        img = img.crop((0, TITLE_BAR_HEIGHT, width, height))
    return img
