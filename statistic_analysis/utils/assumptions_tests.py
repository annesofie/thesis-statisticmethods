import scipy
from scipy import stats
import numpy as np

from .methods import readCsvFile, readFilesReturnFilteredData, normalplot_data


def shapiroWiikNormalityTest(filename, filter):
    '''
    The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.
    :param filename:
    :param filter:
    :return:
    '''
    data_all = readCsvFile(filename)
    data = data_all[filter]
    result_shapiro = scipy.stats.shapiro(data)
    matrix_sw = [
        ['', 'Entries', 'Test statistic', 'p-value'],
        ['Sample Data', len(data) - 1, result_shapiro[0], result_shapiro[1]]
    ]
    return matrix_sw


def normalityTest(filename, filter):
    '''
    Based on D'Agostino and Pearson normality test
        H0: The sample comes from a normal distribution
    Input: Array containing data to be tested
    :return:
     p-value: 2 sided chi squared probability for the hypothesis
    '''

    data_all = readCsvFile(filename)
    data = data_all[filter]
    results = scipy.stats.mstats.normaltest(data)
    matrix_ap = [
        ['', 'Entries', 's^2 + kˆ2', 'p-value'],
        ['Sample Data', len(data) - 1, results[0], results[1]]
    ]
    return matrix_ap


def normalityTest_data(data):
    '''
    Based on D'Agostino and Pearson normality test
        H0: The sample comes from a normal distribution
    Input: Array containing data to be tested
    :return:
     p-value: 2 sided chi squared probability for the hypothesis
    '''
    results = scipy.stats.mstats.normaltest(data)
    matrix_ap = [
        ['', 'Entries', 's^2 + kˆ2', 'p-value'],
        ['Sample Data', len(data) - 1, results[0], results[1]]
    ]
    print(matrix_ap)
    return matrix_ap


def andersonDarlingTest(filename, field):
    '''
    The Anderson-Darling test is a modification of the Kolmogorov- Smirnov test kstest for the null hypothesis
    that a sample is drawn from a population that follows a particular distribution
    :param filename:
    :param field:
    :return:
    '''
    data_all = readCsvFile(filename)
    data = data_all[field]
    anderson_results = scipy.stats.anderson(data, dist='norm')
    matrix_ad = [
        ['', 'Number of entries', 'Test Statistic', 'p-value', 'ALL'],
        ['Sample Data', len(data) - 1, anderson_results[0], anderson_results[1][2], anderson_results]
    ]
    return matrix_ad


def testEqualVariancesTest(filename1, filename2, filter):
    '''
    Tests the null hypothesis that all input samples are from populations with equal variances
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.levene.html
    :param filename1:
    :param filename2:
    :param filter:
    :param center:
            ‘median’ : Recommended for skewed (non-normal) distributions>
            ‘mean’ : Recommended for symmetric, moderate-tailed distributions.
            ‘trimmed’ : Recommended for heavy-tailed distributions.
    :return:
        test statistic
        p-value
    '''
    data1, data2 = readFilesReturnFilteredData(filename1, filename2, filter)
    result_levene = stats.levene(data1, data2, center='mean')
    matrix_levene = [
        ['', 'Number of entries 1', 'Number of entries 2', 'Test Statistic', 'p-value', 'ALL'],
        ['Sample Data', len(data1) - 1, len(data2) - 1, result_levene[0], result_levene[1], result_levene]
    ]
    return matrix_levene


def log10_transform_sample(filename, filter, title):
    '''
    Return the base 10 logarithm of the input array, element-wise.
    :param filename:
    :param filter:
    :return:
        log 10 array
    '''
    data = readCsvFile(filename)
    log_data = np.log10(data[filter])
    normalplot_data(log_data, title)
    normalityTest_data(log_data)


def boxcox_transform_sample(filename, filter, title):
    '''
        Return a positive dataset transformed by a Box-Cox power transformation (log-likelihood function)
    :param filename:
    :param filter:
    :return:
        boxcox : ndarray
            Box-Cox power transformed array.
        maxlog : float, optional
            If the lmbda parameter is None, the second returned argument
            is the lambda that maximizes the log-likelihood function
    '''
    data = readCsvFile(filename)
    boxcox_data = stats.boxcox(data[filter])
    normalplot_data(boxcox_data[0], title)
    normalityTest_data(boxcox_data[0])
    print(boxcox_data[1])