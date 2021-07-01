#################################################################################
#
#             Project Title:  Utilities for CAML Sandbox
#             Author:         Sam Showalter
#             Date:           2021-06-30
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import numpy as np
import os
import sys

#################################################################################
#   Function-Class Declaration
#################################################################################

def convert_to_rgb(val, minval = 0, maxval = 10, reverse = False):
    """
    Borrowed from https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map
    """
    val = val % maxval
    colors = [(0, 0, 255), (0, 100, 0), (255, 0, 0)]
    if reverse:
        colors = list(reversed(colors))
    # `colors` is a series of RGB colors delineating a series of
    # adjacent linear color gradients between each pair.

    # Determine where the given value falls proportionality within
    # the range from minval->maxval and scale that fractional value
    # by the total number in the `colors` palette.
    i_f = float(val-minval) / float(maxval-minval) * (len(colors)-1)

    # Determine the lower index of the pair of color indices this
    # value corresponds and its fractional distance between the lower
    # and the upper colors.
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.

    # Does it fall exactly on one of the color points?
    if f < sys.float_info.epsilon:
        return colors[i]
    else: # Return a color linearly interpolated in the range between it and
          # the following one.
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return (int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1)))

#################################################################################
#   Main Method
#################################################################################



