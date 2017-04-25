import pandas
import scipy
from pandas.tools import plotting
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

local_filepath = '/Users/AnneSofie/Documents/5.klasse/master/statistic_analysis/data/'

file_all_exclude_task4 = 'allParticipantsResultExcludeTask4.csv'
file_all_exclude_task4_exclude_participant = 'allParticipantsResultExcludeTask4ExcludeLastParticipant.csv'
file_all_exclude_task4_exclude_2participants = 'allParticipantsResultExcludeTask4And2Participants.csv'
file_experienced_exclude_task4 = 'experiencedResultExcludeTask4.csv'
file_experienced = 'experiencedResult.csv'
file_non_experienced_exclude_task4 = 'nonExperiencedResultExcludeTask4.csv'
file_non_experienced = 'nonExperiencedResult.csv'


def readCsvFile(filename=None):
    if filename is None:
        filename = 'allParticipantsResult.csv'

    data = pandas.read_csv(local_filepath + filename)
    dimentions = data.shape
    column_names = data.columns

    geom_time = data['geomtasktime']
    meta_time = data['metatasktime']
    total_time = data['totaltime']

    correct_geom = data['correctgeom']
    correct_meta = data['correctmetadata']

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


def pandaDescribe(data):
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


def normalDistributionFit(filename):
    data_all = readCsvFile(filename)
    filter = 'metatasktime'
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
