import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from matplotlib import pyplot as plt

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

DEF_HEAT_COLORMAP = 'inferno'
GRAYSCALE_CONV = np.array([0.2989, 0.5870, 0.1140])


def fade_image(image, fade_level):
    """
    Fades the given image by reducing its brightness by a given amount.
    :param Image.Image image: the original image frame.
    :param float fade_level: the fade level in [0,1], where 0 corresponds to the original image, 1 to a black surface.
    :rtype: Image.Image
    :return: the faded image.
    """
    return ImageEnhance.Brightness(image).enhance(1 - fade_level)


def resize_image_canvas(img, size, color=0):
    """
    Places the given image in a blank canvas of a desired size without resizing the original image itself.
    :param Image.Image img: the original for which to resize the canvas.
    :param (int,int) size: the desired size of the image.
    :param int or str or tuple[int] color: the "blank" color to initialize the new image.
    :rtype: Image.Image
    :return: a new image placed in the blank canvas or the original image, if it's already of the desired size.
    """
    if img.size != size:
        new_img = Image.new(img.mode, size, color)
        new_img.paste(img)
        img = new_img
    return img


def get_max_size(images):
    """
    Gets the maximum image size among the given images.
    :param list[Image.Image] images: the images from which to get the max size.
    :rtype: (int, int)
    :return: a tuple containing the maximum image size among the given images.
    """
    max_size = [0, 0]
    for img in images:
        max_size[0] = max(max_size[0], img.width)
        max_size[1] = max(max_size[1], img.height)
    return tuple(max_size)


def get_mean_image(images, canvas_color=0):
    """
    Gets an image representing the mean of the given images.
    See: https://stackoverflow.com/a/17383621
    :param list[Image.Image] or np.ndarray images: the images to be converted.
    :param int or str or tuple[int] canvas_color: the "blank" color to fill in the canvas of out-of-size frames.
    :rtype: Image.Image
    :return: an image representation of the pixel-mean between the given images.
    """
    if isinstance(images[0], Image.Image):
        max_size = get_max_size(images)
        images = np.array(
            [np.array(resize_image_canvas(img, max_size, canvas_color), dtype=np.float) for img in images])
    return Image.fromarray(np.array(np.round(images.mean(axis=0)), dtype=np.uint8))


def get_variance_heatmap(images, normalize=True, std_dev=False, color_map=DEF_HEAT_COLORMAP, canvas_color=0):
    """
    Gets the variance of the given images as a heatmap image.
    See: https://stackoverflow.com/a/59537945;
        https://stackoverflow.com/a/17383621
    :param list[Image.Image] or np.ndarray images: the images to be converted.
    :param bool normalize: whether to normalize the image values before computing the variance.
    :param str color_map: the name of the matplotlib colormap to produce the heatmap.
    :param bool std_dev: whether to compute standard deviation instead of variance.
    :param int or str or tuple[int] canvas_color: the "blank" color to fill in the canvas of out-of-size frames.
    :rtype: Image.Image
    :return: an image representation of the pixel-variance between the given images.
    """
    if isinstance(images[0], Image.Image):
        max_size = get_max_size(images)
        images = np.array(
            [np.array(resize_image_canvas(img, max_size, canvas_color), dtype=np.float) for img in images])

    # first convert images to grayscale
    images = np.dot(images[..., :3], GRAYSCALE_CONV)

    # get (normalized) variance
    norm_factor = (2 ** 8) if normalize else 1
    images /= norm_factor
    img = ((images.std(axis=0) if std_dev else images.var(axis=0)) * norm_factor).astype(np.uint8)

    # convert to heatmap
    colormap = plt.get_cmap(color_map)
    img = (colormap(img) * 2 ** 8).astype(np.uint8)[..., :3]

    return Image.fromarray(img)


def overlay_square(image, x, y, width, height, color):
    """
    Draws a semi-transparent colored square over the given image.
    :param Image.Image image: the original image.
    :param int x: the left location relative to the original image where the square should be drawn.
    :param int y: the top location relative to the original image where the square should be drawn.
    :param int width: the width of the rectangle to be drawn.
    :param int height: the height of the rectangle to be drawn.
    :param list[int] color: the color of the square to be drawn, in the RGB (assumes fully opaque) or RGBA format.
    :rtype: Image.Image
    :return: the original image with a square overlaid.
    """
    # make blank, fully transparent image the same size and draw a semi-transparent colored square on it
    alpha = color[3] if len(color) >= 4 else 255
    color = color[:3]
    overlay = Image.new('RGBA', image.size, color + [0])
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(((x, y), (x + width, y + height)), fill=color + [alpha])

    # alpha composite the two images together
    return Image.alpha_composite(image, overlay)
