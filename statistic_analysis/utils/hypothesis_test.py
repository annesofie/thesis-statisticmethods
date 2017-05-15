from scipy import stats
import numpy as np
from .methods import readFileReturnFilteredData, readFilesReturnFilteredData, readCsvFile
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison


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


def two_sample_ttest(filename1, filename2, filter=None, equal_var=True):
    """
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
    """
    if filter is None:
        data1 = filename1
        data2 = filename2
    else:
        data1, data2 = readFilesReturnFilteredData(filename1, filename2, filter)

    result_ttest = stats.ttest_ind(data1, data2, 0, equal_var)
    return result_ttest


def two_sample_ttest_descriptive_statistic(mean1, std1, n1, mean2, std2, n2, equal_var=True):
    """
        T-test for means of two independent samples from descriptive statistics.
        This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values.
    :param data1:
    :param data2:
    :return:

    """
    result_test = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var)
    return result_test


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


def testing_oneway_ANOVA(filename1, filename2, filename3, filter=None):
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
    if filter is None:
        data1 = filename1
        data2 = filename2
        data3 = filename3
    else:
        data1, data2 = readFilesReturnFilteredData(filename1, filename2, filter)
        data3 = readFileReturnFilteredData(filename3, filter)

    result_anova = stats.f_oneway(data1, data2, data3)
    return result_anova


def test_wilcoxon_rank_sum(filename1, filename2, filter):
    """
        Tests the null hypothesis that two sets of measurements are drawn from the same distribution.
        The alternative hypothesis is that values in one sample are more likely to be larger than the values in the other sample.
    :param filename1:
    :param filename2:
    :param filter:
    :return:

    """


def test_mannwhitneyu(filename1, filename2, filter=None, use_continuity=True, alternative=None):
    """
        Computes the Mann-Whitney rank test on samples x and y.
    :param use_continuity : bool, optional
        Whether a continuity correction (1/2.) should be taken into account. Default is True.
    :param alternative : None (deprecated), ‘less’, ‘two-sided’, or ‘greater’
        Whether to get the p-value for the one-sided hypothesis (‘less’ or ‘greater’) or for the two-sided hypothesis (‘two-sided’).
        Defaults to None, which results in a p-value half the size of the ‘two-sided’ p-value and a different U statistic.
        The default behavior is not the same as using ‘less’ or ‘greater’: it only exists for backward compatibility and is deprecated.
    :return:
        statistic : float
            The Mann-Whitney U statistic, equal to min(U for x, U for y) if alternative is equal to None (deprecated;
            Exists for backward compatibility), and U for y otherwise.
        pvalue : float
            p-value assuming an asymptotic normal distribution.
            One-sided or two-sided, depending on the choice of alternative.
    """
    if filter is None:
        data1 = filename1
        data2 = filename2
    else:
        data1, data2 = readFilesReturnFilteredData(filename1, filename2, filter)

    result_mannw = stats.mannwhitneyu(data1, data2, use_continuity, alternative)
    return result_mannw


def test_kruskalwallis(filename1, filename2, filename3, filter=None):
    """
        Tests the null hypothesis that the population median of all of the groups are equal.
        It is a non-parametric version of ANOVA
    :param filename1:
    :param filename2:
    :param filename3:
    :param filter:
    :return:
        H-statistic : float
            The Kruskal-Wallis H statistic, corrected for ties
        p-value : float
            The p-value for the test using the assumption that H has a chi square distribution
    """
    if filter is None:
        data1 = filename1
        data2 = filename2
        data3 = filename3
    else:
        data1, data2 = readFilesReturnFilteredData(filename1, filename2, filter)
        data3 = readFileReturnFilteredData(filename3, filter)

    result_kwallis = stats.mstats.kruskalwallis(data1, data2, data3)
    return result_kwallis


def oneway_anova_posthoc_tukey(filename1, data_filter, signlev, transformeddata=None):
    """
        calculate all pairwise comparisons with TukeyHSD confidence intervals
        this is just a wrapper around tukeyhsd method of MultiComparison
    :param filename1:
    :param filaneme2:
    :param filename3:
    :param filter:
    :return:

    """
    data = readCsvFile(filename1)
    if transformeddata is None:
        data1 = data['time']
    else:
        data1 = transformeddata
    mc = pairwise_tukeyhsd(data1, data['task'], signlev)
    print(mc)
    print(mc.groupsunique)
