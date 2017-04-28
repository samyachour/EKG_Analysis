import matplotlib.pyplot as plt

def plot(y, title="Signal", xLab="Index * 0.003s", yLab="mV", size=(9.7,6)):
    fig = plt.figure(figsize=size) # I used figures to customize size
    ax = fig.add_subplot(111)
    ax.plot(y)
    ax.set_title(title)
    # fig.savefig('/Users/samy/Downloads/{0}.png'.format(self.name))
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    plt.show()

def multiplot(data, graph_names):
    #plot multiple lines in one graph
    # input:
    #   data = list of data to plot
    #   graph_names = list of record names to show in the legend
    for l in data:
        plt.plot(l)
    plt.legend(graph_names)
    plt.show()
    
def plotRPeaks(signal):
    fig = plt.figure(figsize=(9.7, 6)) # I used figures to customize size
    ax = fig.add_subplot(111)
    ax.plot(signal.data)
    # ax.axhline(self.baseline)
    ax.plot(*zip(*signal.RPeaks), marker='o', color='r', ls='')
    ax.set_title(signal.name)
    # fig.savefig('/Users/samy/Downloads/{0}.png'.format(self.name))
    plt.show()
    
def plotCoords(data, coords):
    fig = plt.figure(figsize=(9.7, 6)) # I used figures to customize size
    ax = fig.add_subplot(111)
    ax.plot(data)
    ax.plot(*zip(*coords), marker='o', color='r', ls='')
    plt.show()
    
def plotBins(bins, recordTitle=""):
    fig = plt.figure(figsize=(9.7, 6))
    ax = fig.add_subplot(111)
    rects1 = ax.bar(0.5, bins[0], color='r')
    rects2 = ax.bar(1.5, bins[1], color='b')
    rects3 = ax.bar(2.5, bins[2], color='g')
    ax.legend((rects1[0], rects2[0], rects3[0]), ('bin 1', 'bin 2', 'bin 3'))
    ax.set_ylabel('Bin percent')
    ax.set_xlabel('Bins')
    ax.set_title('RR Interval bins' + recordTitle)
    plt.show()
    
     
    