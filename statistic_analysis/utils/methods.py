import pandas
import scipy
from pandas.tools import plotting
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pylab

local_filepath = '/Users/AnneSofie/Documents/5.klasse/master/thesis-statisticmethods/statistic_analysis/data/'


def readCsvFile(filename=None):
    if filename is None:
        filename = 'allParticipantsResult.csv'

    data = pandas.read_csv(local_filepath + filename)
    return data


def readFileReturnFilteredData(filename, filter):
    data_all = readCsvFile(filename)
    return data_all[filter]


def readFilesReturnFilteredData(filename1, filename2, filter):
    data_all1 = readCsvFile(filename1)
    data_all2 = readCsvFile(filename2)
    data1 = data_all1[filter]
    data2 = data_all2[filter]

    return [data1, data2]


def calculateMean(data):
    geom_5_mean = data[data['correctgeom'] == 5].mean()  # Mean for results where correct geom equals 5
    return data.mean()


def pandaDescribe(filename, filter):
    data = readFileReturnFilteredData(filename, filter)
    return pandas.DataFrame.describe(data)  # Calculates count, mean, std, min, max


def createScatterMatrix(filename1, filename2, filename3, field):
    data1 = readCsvFile(filename1)
    data1 = data1[field]
    data2 = readCsvFile(filename2)
    data2 = data2[field]
    data3 = readCsvFile(filename3)
    data3 = data3[field]
    df = pandas.DataFrame({'all': data1, 'experienced': data2, 'non-experienced': data3})
    title = " Scatterplot  Filter: " + field
    plotting.scatter_matrix(df)
    plt.suptitle(title)
    plt.show()
    # return plotting.scatter_matrix(data[['geomtasktime', 'metatasktime', 'totaltime']])


def pairedTest(filename1, filename2):
    data1 = readCsvFile(filename1)
    data2 = readCsvFile(filename2)

    # return stats.ttest_ind(data1['totaltime'], data2['totaltime'])
    return stats.ttest_1samp(data1['correctgeom'] - data1['correctmetadata'], 0)  # 1 sample t-test


def createHistogram(filename, filter, xlabel='', ylabel='', title=''):
    data = readFileReturnFilteredData(filename, filter)
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def createMeanPlot(mean1, mean2, mean3, xlabel='', ylabel='', title=''):
    plt.plot([1, 2, 3], [mean1, mean2, mean3], 'ro')
    plt.axis([0, 3, 100, 200])
    plt.margins(0.05)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def normalplot(filename, filter, title, xlabel):
    data = readCsvFile(filename)
    fit = stats.norm.pdf(data[filter], np.mean(data[filter]), np.std(data[filter]))
    plt.plot(data[filter], fit, '-o')
    plt.hist(data[filter], normed=True)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


def normalplot_data(data, title):
    fit = stats.norm.pdf(data, np.mean(data), np.std(data))
    plt.plot(data, fit, '-o')
    plt.hist(data, normed=True)
    plt.title(title)
    plt.show()


def normalDistributionFit(filename, filter):
    data_all = readCsvFile(filename)
    data = data_all[filter]

    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)

    # Plot the histogram.
    plt.hist(data)

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    print(xmin)
    print(xmax)
    x = np.linspace(xmin, xmax)
    p = norm.pdf(x, mu, std)
    p *= 1000
    plt.plot(x, p, 'r', linewidth=2)
    title = filter + " Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)

    plt.show()


def create_bar_chart(filename, filter):
    data = readCsvFile(filename)
    N = len(data)
    x = range(N)
    y = data[filter]
    print(len(data) - 1)
    colors = []
    for value in data['task_id']:
        if value == 1:
            colors.append('orange')
        elif value == 2:
            colors.append('b')
        else:
            colors.append('g')

    plt.bar(x, y, align='center', color=colors, alpha=0.7)
    # plt.xticks(x, data['participant_age'])
    plt.ylabel(filter)
    plt.xlabel('Participants')

    orange_patch = mpatches.Patch(color='orange', alpha=0.7, label='Task 1')
    blue_patch = mpatches.Patch(color='b', alpha=0.7, label='Task 2')
    green_patch = mpatches.Patch(color='g', alpha=0.7, label='Task 3')
    plt.legend(handles=[orange_patch, blue_patch, green_patch])
    plt.show()


def create_bar_chart_data_x(filename, filter):
    data = readCsvFile(filename)
    N = len(data)
    y = range(N)
    x = data[filter]
    print(len(data) - 1)
    colors = []
    for value in data['task_id']:
        if value == 1:
            colors.append('orange')
        elif value == 2:
            colors.append('b')
        else:
            colors.append('g')

    plt.bar(x, y, align='center', color=colors, alpha=0.7)
    # plt.xticks(x, data['participant_age'])
    plt.xlabel(filter)
    plt.ylabel('Participants')

    orange_patch = mpatches.Patch(color='orange', alpha=0.7, label='Task 1')
    blue_patch = mpatches.Patch(color='b', alpha=0.7, label='Task 2')
    green_patch = mpatches.Patch(color='g', alpha=0.7, label='Task 3')
    plt.legend(handles=[orange_patch, blue_patch, green_patch])
    plt.show()


def create_bar_chart_survey(filename, filter):
    data = readCsvFile(filename)
    N = len(data)
    x = range(N)
    y = data[filter]
    print(y)
    colors = []
    for value in data['task_id']:
        if value == 1:
            colors.append('orange')
        elif value == 2:
            colors.append('b')
        elif value == 3:
            colors.append('g')
        else:
            colors.append('yellow')

    plt.bar(x, y, align='center', color=colors, alpha=0.7)
    plt.xticks(x, data['difficulty'])
    plt.ylabel(filter)
    plt.xlabel('Difficulty')

    orange_patch = mpatches.Patch(color='orange', alpha=0.7, label='Task 1')
    blue_patch = mpatches.Patch(color='b', alpha=0.7, label='Task 2')
    green_patch = mpatches.Patch(color='g', alpha=0.7, label='Task 3')
    plt.legend(handles=[orange_patch, blue_patch, green_patch])
    plt.show()


def create_probplot(filename, filter, title):
    '''
    Generates a probability plot of sample data against the quantiles of a specified theoretical distribution (the normal distribution by default)
    :param filename:
    :param filter:
    :param title:
    :return:
    '''
    data = readCsvFile(filename)
    scipy.stats.probplot(data[filter], dist='norm', plot=pylab)
    pylab.title(title)
    pylab.show()
