###############################################################################
#
#             Project Title:  Environment and Argparse Utilities for Cameleon Env
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import pickle as pkl
import hickle as hkl
from operator import add

#################################################################################
#   Function-Class Declaration
#################################################################################

def _write_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pkl.dump(obj, file)

def _read_pkl(filename):
    with open(filename, 'rb') as file:
        return pkl.load(file)


def _write_hkl(obj, filename):
    # with open(filename, 'wb') as file:
    hkl.dump(obj, filename, mode = 'w',
             compression='gzip')

def _read_hkl(filename):
    return hkl.load(filename)

def _tup_equal(t1,t2):
    """Check to make sure to tuples are equal

    :t1: Tuple 1
    :t2: Tuple 2
    :returns: boolean equality

    """
    if t1 is None or t2 is None:
        return False
    return (t1[0] == t2[0]) and (t1[1] == t2[1])

def _tup_add(t1, t2):
    """Add two tuples

    :t1: Tuple 1
    :t2: Tuple 2

    :returns: Tuple sum

    """
    return tuple(map(add,t1,t2))

def _tup_mult(t1, t2):
    """Multiply tuples

    :t1: Tuple 1
    :t2: Tuple 2

    :returns: Tuple sum

    """
    return (t1[0]*t2[0],t1[1]*t2[1])

#######################################################################
# Main
#######################################################################

