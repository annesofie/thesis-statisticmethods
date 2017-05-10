from scipy import stats
from .methods import readFileReturnFilteredData, readFilesReturnFilteredData

def zscore_calculation(filename, filter):
    '''
        Calculates the z score of each value in the sample, relative to the sample mean and standard deviation
    :param filename:
    :param filter:
    :return:
    z-zcore
    '''
    data = readFileReturnFilteredData(filename, filter)
    return stats.zscore(data, ddof=1)

def two_sample_ttest(filename1, filename2, filter):
    '''
    Two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values

    :param filename1:
    :param filename2:
    :param filter:
    :param equal_var
        If True (default), perform a standard independent 2 sample test that assumes equal population variances [R643].
        If False, perform Welch’s t-test, which does not assume equal population variance [R644].
    :return:
        t-statistic
        p-value
    '''
    data1, data2 = readFilesReturnFilteredData(filename1, filename2, filter)
    result_ttest = stats.ttest_ind(data1, data2)
    return result_ttest


def testingPopulationMean(filename, filter):
    '''
    Two sided test for the null hypothesis that the expected value (mean) is the given population mean

    returns
        statistic : float or array
            t-statistic
        pvalue : float or array
            two-tailed p-value
    '''
    data = readFileReturnFilteredData(filename, filter)

    res_geom2 = stats.ttest_1samp(data['geomtasktime'], 156)
    res_meta2 = stats.ttest_1samp(data['metatasktime'], 86)
    res_total2 = stats.ttest_1samp(data['totaltime'], 242)


def testing_oneway_ANOVA(filename1, filename2, filename3, filter):
    '''
        The null hypothesis that two or more groups have the same population mean.
        The test is applied to samples from two or more groups, possibly with differing sizes.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    :param filename1:
    :param filename2:
    :param filename3:
    :param filter:
    :return:
        F-value
        p-value
    '''
    data1, data2 = readFilesReturnFilteredData(filename1, filename2, filter)
    data3 = readFileReturnFilteredData(filename3, filter)
    result_anova = stats.f_oneway(data1, data2, data3)
    return result_anova

def test_wilcoxon_rank_sum(filename1, filename2, filter):
    '''
        Tests the null hypothesis that two sets of measurements are drawn from the same distribution.
        The alternative hypothesis is that values in one sample are more likely to be larger than the values in the other sample.
    :param filename1:
    :param filename2:
    :param filter:
    :return:

    '''