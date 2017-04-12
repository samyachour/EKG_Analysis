import matplotlib.pyplot as plt

def plot(y, title="Signal", xLab="Index * 0.003s"):
    fig = plt.figure(figsize=(9.7, 6)) # I used figures to customize size
    ax = fig.add_subplot(111)
    ax.plot(y)
    ax.set_title(title)
    # fig.savefig('/Users/samy/Downloads/{0}.png'.format(self.name))
    ax.set_ylabel("mV")
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