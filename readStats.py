import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
from scipy.stats import norm
import yaml
import sys
import getopt

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
        flat window will produce a moving average smoothing.

        output:
        the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also:

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def readAndPlot(data, diffActorsStats, labelPrefix, filePath):
    nSamples = data['NSamples']
    stats = data['Stats']

    generateLut = labelPrefix != 'DifferentActors';

    for name, results in stats.iteritems():

        nBins = 128

        value = results['values']
        rangeDiffActors = []
        diffActorsValue = []
        if generateLut:
            diffActorsResults = diffActorsStats['Stats'][name]
            diffActorsValue = diffActorsResults['values']
            rangeDiffActors = (diffActorsResults['min'], diffActorsResults['max'])

        avg = results['avg']
        std = results['stdev']
        maxVal = results['max']
        minVal = results['min']
        plt.clf()
        if not generateLut:
            histoRange = (minVal, maxVal)
        else:
            histoRange = rangeDiffActors
        histo, bins, patches = plt.hist(value, nBins, histoRange)
        plt.xlabel(labelPrefix + '_' + name)
        plt.ylabel('Probability')
        prec='.5f'
        plt.title(name + ': min = ' + format(minVal, prec)  + ', max = ' + format(maxVal, prec) + '\n avg = ' + format(avg, prec) + ', std = ' + format(std, prec))
        plt.grid(True)
        plt.legend(loc='upper right')
        figpath = filePath + labelPrefix + '_' + name + '_figure.png'
        print 'saving ' + figpath
        plt.savefig(figpath)

        if generateLut:
            histoDiff,binsDiff,patchesDiff = plt.hist(diffActorsValue,nBins, histoRange)
            histo = smooth(histo)
            histoDiff = smooth(histoDiff)
            step = (histoRange[1] - histoRange[0]) / nBins
            binValue = histoRange[0]
            yValues=np.empty(nBins)
            xValues=np.empty(nBins)
            for i in xrange(0,nBins):
                eps=10e-12
                a = histo[i] + histoDiff[i]
                if a > eps:
                    yValues[i] = histo[i] / a
                else:
                    yValues[i] = 0
                yValues[i] = -np.log(yValues[i] + eps)

                xValues[i] = binValue
                binValue += step

            plt.clf()
            plt.xlabel(labelPrefix + '_' + name + '_transferLUT')
            plt.plot(xValues, yValues)
            figpath = filePath + labelPrefix + '_' + name + '_transferLUT_figure.png'
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

    diffActorsStats=fileContent['DifferentActors']
    for category, statistics in fileContent.iteritems():
        readAndPlot(statistics, diffActorsStats, category, filePath)


if __name__ == "__main__":
    main(sys.argv[1:])
