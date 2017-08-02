import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import yaml
import sys
import getopt


def readAndPlot(data, labelPrefix, filePath):
    nSamples = data['NSamples']
    stats = data['Stats']

    for name, value in stats.iteritems():

        avg = sum(value) / float(len(value))
        std = np.std(value)
        plt.clf()
        plt.hist(value, 128)
        plt.xlabel(labelPrefix + '_' + name)
        plt.ylabel('Probability')
        plt.title(r'$\mathrm{Histogram\ of\ ' + name + ':}\ avg = ' + str(avg) + ',\ std = ' + str(std) + '$')
        plt.grid(True)
        plt.legend(loc='upper right')
        figpath = filePath + labelPrefix + '_' + name + '_figure.png'
        print 'saving ' + figpath
        plt.savefig(figpath)

def main(argv):

    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"i:",["ifile="])
        for opt,value in opts:
            if opt == "-i":
                inputfile = value

    except getopt.GetoptError:
        print 'readStats.py -i <inputfile>'
        sys.exit(1)

    fileContent=''
    with open(inputfile, 'r') as stream:
        try:
            fileContent = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    foundSlash = inputfile.rfind('/')
    filePath = ''
    if foundSlash != -1:
        filePath = inputfile[:foundSlash + 1]
    differentActors=fileContent['DifferentActor']
    sameActors=fileContent['SameActor']

#print 'dumping stats for different actors'
#    print '-----------------------'
    readAndPlot(differentActors, 'diffActors', filePath)
    #    print 'dumping stats for same actors'
    #print '-----------------------'
    readAndPlot(sameActors, 'sameActor', filePath)

#plt.title("Gaussian Histogram")
#plt.xlabel("Value")
#plt.ylabel("Frequency")

#fig = plt.gcf()

#plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')


if __name__ == "__main__":
    main(sys.argv[1:])
