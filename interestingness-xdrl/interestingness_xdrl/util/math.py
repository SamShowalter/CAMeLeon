import numpy
import numpy as np

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def get_variation_ratio(dist):
    """
    Gets the statistical dispersion of a given nominal distribution in [0,1]. The larger the ratio (-> 1), the more
    differentiated or dispersed the data are. The smaller the ratio (0 <-), the more concentrated the distribution is.
    See: https://en.wikipedia.org/wiki/Variation_ratio
    :param np.ndarray dist: the nominal distribution.
    :rtype: float
    :return: a measure of the dispersion of the data in [0,1].
    """
    total = np.sum(dist)
    mode = np.max(dist)
    return 1. - mode / total


def get_distribution_evenness(dist):
    """
    Gets the evenness of a given nominal distribution as the normalized true diversity in [0,1]. It has into account the
    number of different categories one would expect to find in the distribution, i.e. it handles 0 entries.
    The larger the ratio (-> 1), the more differentiated or even the data are.
    The smaller the ratio (0 <-), the more concentrated the distribution is, i.e., the more uneven the data are.
    See: https://en.wikipedia.org/wiki/Species_evenness
    See: https://en.wikipedia.org/wiki/Diversity_index#Shannon_index
    :param np.ndarray dist: the nominal distribution.
    :rtype: float
    :return: a measure of the evenness of the data in [0,1].
    """
    num_expected_elems = len(dist)
    nonzero = np.nonzero(dist)[0]
    if len(nonzero) == 0:
        return 1. / np.log(num_expected_elems)
    if len(nonzero) == 1:
        return 0.
    dist = np.array(dist)[nonzero]
    total = np.sum(dist)
    dist = np.true_divide(dist, total)
    in_dist = [p * np.log(p) for p in dist]
    return - np.sum(in_dist) / np.log(num_expected_elems)


def get_outliers_double_mads(data, thresh=3.5):
    """
    Identifies outliers in a given data set according to the data-points' "median absolute deviation" (MAD), i.e.,
    measures the distance of all points from the median in terms of median distance.
    From answer at: https://stackoverflow.com/a/29222992
    :param np.ndarray data: the data from which to extract the outliers.
    :param float thresh: the z-score threshold above which a data-point is considered an outlier.
    :rtype: np.ndarray
    :return: an array containing the indexes of the data that are considered outliers.
    """
    # warning: this function does not check for NAs nor does it address
    # issues when more than 50% of your data have identical values
    m = np.median(data)
    abs_dev = np.abs(data - m)
    left_mad = np.median(abs_dev[data <= m])
    right_mad = np.median(abs_dev[data >= m])
    y_mad = left_mad * np.ones(len(data))
    y_mad[data > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[data == m] = 0
    return np.where(modified_z_score > thresh)[0]


def get_outliers_dist_mean(data, std_devs=2., above=True, below=True):
    """
    Identifies outliers according to distance of a number of standard deviations to the mean.
    :param np.ndarray data: the data from which to extract the outliers.
    :param float std_devs: the number of standard deviations above/below which a point is considered an outlier.
    :param bool above: whether to consider outliers above the mean.
    :param bool below: whether to consider outliers below the mean.
    :rtype: np.ndarray
    :return: an array containing the indexes of the data that are considered outliers.
    """
    mean = np.mean(data)
    std = np.std(data)
    outliers = [False] * len(data)
    if above:
        outliers |= data >= mean + std_devs * std
    if below:
        outliers |= data <= mean - std_devs * std
    return np.where(outliers)[0]


def get_jensen_shannon_divergence(dist1, dist2):
    """
    Computes the Jensen-Shannon divergence between two probability distributions. Higher values (close to 1) mean that
    the distributions are very dissimilar while low values (close to 0) denote a low divergence, similar distributions.
    See: https://stackoverflow.com/a/40545237
    See: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    Ref: Lin J. 1991. "Divergence Measures Based on the Shannon Entropy".
        IEEE Transactions on Information Theory. (33) 1: 145-151.
    Input must be two probability distributions of equal length that sum to 1.
    :param np.ndarray dist1: the first probability distribution.
    :param np.ndarray dist2: the second probability distribution.
    :rtype: float
    :return: the divergence between the two distributions in [0,1].
    """
    assert dist1.shape == dist2.shape, 'Distribution shapes do not match'

    def _kl_div(a, b):
        return np.nansum(a * np.log2(a / b))

    m = 0.5 * (dist1 + dist2)
    return 0.5 * (_kl_div(dist1, m) + _kl_div(dist2, m))


def get_pairwise_jensen_shannon_divergence(dist1, dist2):
    """
    Computes the pairwise Jensen-Shannon divergence between two probability distributions. This corresponds to the
    un-summed JSD, i.e., the divergence according to each component of the given distributions. Summing up the returned
    array yields the true JSD between the two distributions.
    See: https://stackoverflow.com/a/40545237
    See: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    Input must be two probability distributions of equal length that sum to 1.
    :param np.ndarray dist1: the first probability distribution.
    :param np.ndarray dist2: the second probability distribution.
    :rtype: np.ndarray
    :return: an array the same size of the distributions with the divergence between each component in [0,1].
    """
    assert dist1.shape == dist2.shape, 'Distribution shapes do not match'

    def _kl_div(a, b):
        return a * np.log2(a / b)

    m = 0.5 * (dist1 + dist2)
    return 0.5 * (_kl_div(dist1, m) + _kl_div(dist2, m))


def get_diff_means(mean1, std1, n1, mean2, std2, n2):
    """
    Gets the difference of the given sample means (mean1 - mean2).
    See: https://stattrek.com/sampling/difference-in-means.aspx
    :param float mean1: the first mean value.
    :param float std1: the first mean's standard deviation.
    :param int n1: the first mean's count.
    :param float mean2: the first mean value.
    :param float std2: the first mean's standard deviation.
    :param int n2: the first mean's count.
    :rtype: (float, float, int)
    :return: a tuple containing the differences of the mean, standard deviation and number of elements.
    """
    return \
        mean1 - mean2, \
        np.sqrt((std1 * std1) / n1 + (std2 * std2) / n2).item(), \
        n1 - n2


def save_list_csv(data_list, file_path, sep_char=','):
    """
    Saves a CSV file containing all the elements in the given list.
    :param list data_list: the list of elements to be saved.
    :param str file_path: the path to the CSV file in which to save the list.
    :param str sep_char: the character used to separate elements in the CSV file.
    :return:
    """
    with open(file_path, 'w') as file:
        file.write(sep_char.join([str(elem) for elem in data_list]))


def gaussian_kl_divergence(mu1, mu2, sigma_1, sigma_2):
    """
    Computes the Kullback-Leibler divergence (KL-D) between one Gaussian distribution and another Gaussian distribution
    (or set of distributions). Distributions assumed to have diagonal covariances.
    See: https://stackoverflow.com/q/44549369
    :param np.ndarray mu1: the mean vector of the first distribution.
    :param np.ndarray mu2: the mean vector of the second distribution (or set of distributions).
    :param np.ndarray sigma_1: the covariance diagonal of the first distribution.
    :param np.ndarray sigma_2: the covariance diagonal of the second distribution (or set of distributions).
    :rtype: float
    :return: the KL divergence between the two distributions.
    """
    if len(mu2.shape) == 2:
        axis = 1
    else:
        axis = 0
    mu_diff = mu2 - mu1
    return 0.5 * (np.log(np.prod(sigma_2, axis=axis) / np.prod(sigma_1))
                  - mu1.shape[0] + np.sum(sigma_1 / sigma_2, axis=axis)
                  + np.sum(mu_diff * 1 / sigma_2 * mu_diff, axis=axis))


def gaussian_entropy(sigma):
    """
    Computes the entropy of a multivariate Gaussian distribution with the given covariance.
    See: https://sgfin.github.io/2017/03/11/Deriving-the-information-entropy-of-the-multivariate-gaussian/
    :param np.ndarray sigma: the covariance matrix of the Gaussian distribution. If a 1-dimensional array is provided,
    it is assumed the covariance is diagonal corresponding to the given vector.
    :return: the entropy of the distribution.
    """
    n = sigma.shape[0]
    det = np.linalg.det(sigma) if len(sigma.shape) == 2 else np.prod(sigma)
    return n / 2 * np.log(2 * np.pi * np.e) + 0.5 * np.log(det)
