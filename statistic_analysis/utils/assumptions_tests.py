import scipy
from scipy import stats

from utils.methods import readCsvFile, readFilesReturnFilteredData


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


def andersonDarlingTest(filename, field):
    data_all = readCsvFile(filename)
    data = data_all[field]
    anderson_results = scipy.stats.anderson(data)
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
