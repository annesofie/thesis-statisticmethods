import pandas
import scipy
from pandas.tools import plotting
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

local_filepath = '/Users/AnneSofie/Documents/5.klasse/master/thesis-statisticmethods/statistic_analysis/data/'

# ALL
path = 'all_participants/'
file_all_orderby_totaltime = path + 'all_orderby_totaltime.csv'
file_excludetask4 = path + 'allParticipantsResultExcludeTask4.csv'
file_excludetask4_orderby_totaltime = path + 'allExcludeTask4_Orderby_totaltime.csv'
file_excludetask4_orderby_age_filterlessthan2000 = path + 'allExcludeTask4totaltimebiggerthan2000orderbyparticipant__age.csv'
file_excludetask4_orderby_totalcorrect = path + 'allParticipantsResultExcludeTask4orderbytotal_correct_elements.csv'
file_excludetask4_orderby_age = path + 'allParticipantsResultExcludeTask4orderbyparticipant__age.csv'
file_excludetask4_filtertotaltime_orderby_age = path + 'allExcludeTask4totaltimebiggerthan2000orderbyparticipant__age.csv'

# Interupted
path = 'was_interupted_participants/'
file_was_interupted_orderedby_difficulty = path + 'allParticipants_wasinterupted_orderby_difficulty.csv'

# Experienced
path_e = 'experienced_participants/'
path_in = 'inexperienced_participants/'
file_experienced_orderby_totaltime = path_e + 'experiencedResult_orderby_totaltime.csv'
file_experienced_orderby_totaltime_exclude4 = path_e + 'experiencedResultExcludeTask4_orderby_totaltime.csv'
file_experienced_orderby_age = path_e + 'experiencedResult_orderby_participant__age.csv'
file_inexperienced_orderby_totaltime = path_in + 'nonExperiencedResult_orderby_totaltime.csv'
file_inexperienced_orderby_totaltime_exclude4 = path_in + 'inExperiencedResultExcludeTask4_orderby_totaltime.csv'
file_inexperienced_orderby_age = path_in + 'nonExperiencedResult_orderby_participant__age.csv'

#Task
path_1 = 'task1/'
path_2 = 'task2/'
path_3 = 'task3/'
file_task1_orderedby_totaltime = path_1+'oneElementTaskResult_filtertotaltime_totaltime.csv'
file_task1_orderedby_totaltime_experienced = path_1+'experiencedResult_exclude4_task1.csv'
file_task1_orderedby_totaltime_inexperienced = path_1+'inExperiencedResult_exclude4_task1.csv'
file_task2_orderedby_totaltime = path_2+'threeElementTaskResult_filtertotaltime_totaltime.csv'
file_task2_orderedby_totaltime_experienced = path_2+'experiencedResult_exclude4_task2.csv'
file_task2_orderedby_totaltime_inexperienced = path_2+'inExperiencedResult_exclude4_task2.csv'
file_task3_orderedby_totaltime = path_3+'sixElementTaskResult_filtertotaltime_totaltime.csv'
file_task3_orderedby_totaltime_experienced = path_3+'experiencedResult_exclude4_task3.csv'
file_task3_orderedby_totaltime_inexperienced = path_3+'inExperiencedResult_exclude4_task3.csv'



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



def normalplot(filename, filter, title):
    data = readCsvFile(filename)
    fit = stats.norm.pdf(data[filter], np.mean(data[filter]), np.std(data[filter]))
    plt.plot(data[filter], fit, '-o')
    plt.hist(data[filter], normed=True)
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
