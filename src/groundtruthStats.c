#include "detectorCommon.h"
#include <boost/shared_ptr.hpp>
#include <yaml-cpp/yaml.h>

using std::string;

struct StatData
{
    double dw, dx, dy, dhist_chi2, dhist_intersect, dhist_correl;
};

struct ActorToActorKey
{
    std::string actor1;
    std::string actor2;
};

struct ActorToActorKeyCompareLess
{
    bool operator()(const ActorToActorKey& lhs, const ActorToActorKey& rhs) const
    {
        std::string aCat = lhs.actor1 + lhs.actor2;
        std::string bCat = rhs.actor1 + rhs.actor2;
        return aCat < bCat;
    }
};

struct ActorToActorData
{
    std::list<StatData> datas;
};

typedef std::map<ActorToActorKey, ActorToActorData, ActorToActorKeyCompareLess> PerActorStatDataMap;


static void parseArguments(const std::list<string>& args,
                           string *detectionsFile,
                           string *outputFileName)
{
    std::list<string> localArgs = args;

    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "-i");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("A detections file must be specified with the -i switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("-i switch without a file name");
        } else {
            *detectionsFile = *foundInput;
            localArgs.erase(foundInput);
        }
    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "-o");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("An output file must be specified with the -o switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("-o switch without a file name");
        } else {
            *outputFileName = *foundInput;
            localArgs.erase(foundInput);
        }
    }
} // parseArguments

static void compareDetections(const SERIALIZATION_NAMESPACE::DetectionSerialization& a, const SERIALIZATION_NAMESPACE::DetectionSerialization& b,
                              const std::vector<boost::shared_ptr<FStreamsSupport::ifstream> >& modelFiles,
                              const std::vector<int>& modelSizes,
                              StatData* data)
{
    RectD aRect(a.x1, a.y1, a.x2, a.y2);
    RectD bRect(b.x1, b.y1, b.x2, b.y2);

    data->dw = std::abs(aRect.width() - bRect.width());
    data->dx = std::abs((aRect.x1 + aRect.x2) / 2. -  (bRect.x1 + bRect.x2) / 2.);
    data->dy = std::abs((aRect.y1 + aRect.y2) / 2. -  (bRect.y1 + bRect.y2) / 2.);

    std::vector<double> histogramA, histogramB;
    assert(a.fileIndex < (int)modelFiles.size());
    assert(b.fileIndex < (int)modelFiles.size());
    loadHistogramFromFile(modelFiles[a.fileIndex], a.modelIndex, modelSizes, &histogramA);
    loadHistogramFromFile(modelFiles[b.fileIndex], b.modelIndex, modelSizes, &histogramB);

    data->dhist_chi2 = compareHist(histogramA, histogramB, eHistogramComparisonChiSquare);
    data->dhist_correl = compareHist(histogramA, histogramB, eHistogramComparisonCorrelation);
    data->dhist_intersect = compareHist(histogramA, histogramB, eHistogramComparisonIntersection);
}

static void computeData(const string& inputFilename, const string& outputFilename, PerActorStatDataMap& data) {
    SERIALIZATION_NAMESPACE::SequenceSerialization detectionResults;
    std::string detectionFileDirPath = inputFilename;
    {
        std::size_t foundSlash = detectionFileDirPath.find_last_of("/\\");
        if (foundSlash != std::string::npos) {
            detectionFileDirPath = detectionFileDirPath.substr(0, foundSlash);
        }
    }
    {
        FStreamsSupport::ifstream ifile;
        FStreamsSupport::open(&ifile, inputFilename);
        if (!ifile.good()) {
            throw std::invalid_argument("Failed to open " + inputFilename);
        }

        SERIALIZATION_NAMESPACE::read(SERIALIZATION_FILE_FORMAT_HEADER, ifile, &detectionResults);
    }



    std::vector<boost::shared_ptr<FStreamsSupport::ifstream> > modelFiles;

    for (std::size_t i = 0; i < detectionResults.modelFiles.size(); ++i) {
        std::string modelAbsoluteFilePath = detectionFileDirPath + "/" + detectionResults.modelFiles[i];

        boost::shared_ptr<FStreamsSupport::ifstream> ifile(new FStreamsSupport::ifstream);
        FStreamsSupport::open(ifile.get(), modelAbsoluteFilePath);
        if (!ifile->good()) {
            throw std::invalid_argument("Failed to open " + modelAbsoluteFilePath);
        }
        modelFiles.push_back(ifile);
    }

    {
        std::map<int, SERIALIZATION_NAMESPACE::FrameSerialization>::const_iterator it = detectionResults.frames.begin();
        std::map<int, SERIALIZATION_NAMESPACE::FrameSerialization>::const_iterator next = it;
        ++next;
        for (; next != detectionResults.frames.end(); ++it, ++next) {

            for (std::list<SERIALIZATION_NAMESPACE::DetectionSerialization>::const_iterator it2 = it->second.detections.begin(); it2 != it->second.detections.end(); ++it2) {

                // Search the same actor in the next frame
                for (std::list<SERIALIZATION_NAMESPACE::DetectionSerialization>::const_iterator it3 = next->second.detections.begin(); it3 != next->second.detections.end(); ++it3) {
                    ActorToActorKey key;
                    key.actor1 = it2->label;
                    key.actor2 = it3->label;

                    ActorToActorData& actorsData = data[key];

                    StatData statistics;
                    compareDetections(*it2, *it3, modelFiles, detectionResults.histogramSizes, &statistics);
                    actorsData.datas.push_back(statistics);
                }
            }
            
        }
    }
}

struct StatResults
{
#ifdef COMPUTE_STATS

    std::vector<int> statHistogram;
    double statAverage;
    double statMin, statMax;
    double statSvd;
#else
    std::vector<double> results;

#endif

    StatResults()
    :
#ifdef COMPUTE_STATS
    statHistogram(32, 0)
    , statAverage(0)
    , statMin(0)
    , statMax(0)
    , statSvd(0)
#else
    results()
#endif
    {

    }
};

typedef std::map<string, StatResults> StatResultsMap;

struct PerStatResultsData
{
    StatResultsMap results;

    std::size_t nSamples;

    PerStatResultsData()
    : nSamples(0)
    {

    }
};


typedef std::map<string, PerStatResultsData> PerCategoryStatResults;

inline void addToHistogram(std::vector<int>& histo, double value, double minValue, double maxValue)
{
    double range = maxValue - minValue;
    if (range <= 0) {
        return;
    }
    double x = (value - minValue) / range;
    assert(x == x);
    int i = std::floor(x * (double)(histo.size() - 1) + 0.5);
    assert(i >= 0 && i < (int)histo.size());
    ++histo[i];
}

int produceStatsMain(int argc, char** argv)
{
    std::list<string> arguments;
    for (int i = 0; i < argc; ++i) {
        arguments.push_back(string(argv[i]));
    }

    string inputFilename, outputFilename;
    parseArguments(arguments, &inputFilename, &outputFilename);

    FStreamsSupport::ofstream ofile;
    FStreamsSupport::open(&ofile, outputFilename, std::ios_base::out | std::ios_base::trunc);
    if (!ofile) {
        throw std::invalid_argument("Could not open " + outputFilename);
    }

    PerActorStatDataMap data;
    computeData(inputFilename, outputFilename, data);

    static const string catSameActor("SameActor");
    static const string catDiffActor("DifferentActor");

    PerCategoryStatResults results;

    for (PerActorStatDataMap::const_iterator it = data.begin(); it != data.end(); ++it) {

        if (it->second.datas.size() == 0) {
            continue;
        }
        PerStatResultsData* statMap = 0;
        if (it->first.actor1 == it->first.actor2) {
            statMap = &results[catSameActor];
        } else {
            statMap = &results[catDiffActor];
        }
        StatResults& dwData = statMap->results["dw"];
        StatResults& dxData = statMap->results["dx"];
        StatResults& dyData = statMap->results["dy"];
        StatResults& dhistChi2 = statMap->results["dhistChi2"];
        StatResults& dhistCorrel = statMap->results["dhistCorrel"];
        StatResults& dhistIntersect = statMap->results["dhistIntersection"];
        statMap->nSamples += it->second.datas.size();


        for (std::list<StatData>::const_iterator it2 = it->second.datas.begin(); it2 != it->second.datas.end(); ++it2) {
#ifdef COMPUTE_STATS
            dwData.statAverage += it2->dw;
            dwData.statMin = std::min(dwData.statMin, it2->dw);
            dwData.statMax = std::max(dwData.statMax, it2->dw);

            dxData.statAverage += it2->dx;
            dxData.statMin = std::min(dxData.statMin, it2->dx);
            dxData.statMax = std::max(dxData.statMax, it2->dx);

            dyData.statAverage += it2->dy;
            dyData.statMin = std::min(dyData.statMin, it2->dy);
            dyData.statMax = std::max(dyData.statMax, it2->dy);

            dhistChi2.statAverage += it2->dhist_chi2;
            dhistChi2.statMin = std::min(dhistChi2.statMin, it2->dhist_chi2);
            dhistChi2.statMax = std::max(dhistChi2.statMax, it2->dhist_chi2);

            dhistCorrel.statAverage += it2->dhist_correl;
            dhistCorrel.statMin = std::min(dhistCorrel.statMin, it2->dhist_correl);
            dhistCorrel.statMax = std::max(dhistCorrel.statMax, it2->dhist_correl);

            dhistIntersect.statAverage += it2->dhist_intersect;
            dhistIntersect.statMin = std::min(dhistIntersect.statMin, it2->dhist_intersect);
            dhistIntersect.statMax = std::max(dhistIntersect.statMax, it2->dhist_intersect);
#else
            dwData.results.push_back(it2->dw);
            dxData.results.push_back(it2->dx);
            dyData.results.push_back(it2->dy);
            dhistChi2.results.push_back(it2->dhist_chi2);
            dhistCorrel.results.push_back(it2->dhist_correl);
            dhistIntersect.results.push_back(it2->dhist_intersect);
#endif
        }
#ifdef COMPUTE_STATS
        dwData.statAverage /= statMap->nSamples;
        dxData.statAverage /= statMap->nSamples;
        dyData.statAverage /= statMap->nSamples;
        dhistChi2.statAverage /= statMap->nSamples;
        dhistCorrel.statAverage /= statMap->nSamples;
        dhistIntersect.statAverage /= statMap->nSamples;

        for (std::list<StatData>::const_iterator it2 = it->second.datas.begin(); it2 != it->second.datas.end(); ++it2) {
            addToHistogram(dwData.statHistogram, it2->dw, dwData.statMin, dwData.statMax);
            addToHistogram(dxData.statHistogram, it2->dx, dxData.statMin, dxData.statMax);
            addToHistogram(dyData.statHistogram, it2->dy, dyData.statMin, dyData.statMax);
            addToHistogram(dhistChi2.statHistogram, it2->dhist_chi2, dhistChi2.statMin, dhistChi2.statMax);
            addToHistogram(dhistCorrel.statHistogram, it2->dhist_correl, dhistCorrel.statMin, dhistCorrel.statMax);
            addToHistogram(dhistIntersect.statHistogram, it2->dhist_intersect, dhistIntersect.statMin, dhistIntersect.statMax);

            dwData.statSvd += (it2->dw - dwData.statAverage) * (it2->dw - dwData.statAverage);
            dxData.statSvd += (it2->dx - dxData.statAverage) * (it2->dx - dxData.statAverage);
            dyData.statSvd += (it2->dy - dyData.statAverage) * (it2->dy - dyData.statAverage);
            dhistChi2.statSvd += (it2->dhist_chi2 - dhistChi2.statAverage) * (it2->dhist_chi2 - dhistChi2.statAverage);
            dhistCorrel.statSvd += (it2->dhist_correl - dhistCorrel.statAverage) * (it2->dhist_correl - dhistCorrel.statAverage);
            dhistIntersect.statSvd += (it2->dhist_intersect - dhistIntersect.statAverage) * (it2->dhist_intersect - dhistIntersect.statAverage);
        }

        dwData.statSvd = std::sqrt(dwData.statSvd / statMap->nSamples);
        dxData.statSvd = std::sqrt(dxData.statSvd / statMap->nSamples);
        dyData.statSvd = std::sqrt(dyData.statSvd / statMap->nSamples);
        dhistChi2.statSvd = std::sqrt(dhistChi2.statSvd / statMap->nSamples);
        dhistCorrel.statSvd = std::sqrt(dhistCorrel.statSvd / statMap->nSamples);
        dhistIntersect.statSvd = std::sqrt(dhistIntersect.statSvd / statMap->nSamples);
#endif
    }

    YAML::Emitter em;
    {
        em << YAML::BeginMap;
        for (PerCategoryStatResults::const_iterator it = results.begin(); it != results.end(); ++it) {
            em << YAML::Key << it->first;
            em << YAML::Value;
            const PerStatResultsData& resultsData = it->second;
            em << YAML::BeginMap;
            em << YAML::Key << "NSamples" <<  YAML::Value << resultsData.nSamples;

            em << YAML::Key << "Stats" << YAML::Value;
            em << YAML::BeginMap;
            for (StatResultsMap::const_iterator it2 = resultsData.results.begin(); it2 != resultsData.results.end(); ++it2) {
                em << YAML::Key << it2->first;
                em << YAML::Value;
#ifdef COMPUTE_STATS
                em << YAML::BeginMap;
                em << YAML::Key;
                em << "min";
                em << YAML::Value;
                em << it2->second.statMin;

                em << YAML::Key;
                em << "max";
                em << YAML::Value;
                em << it2->second.statMax;

                em << YAML::Key;
                em << "svd";
                em << YAML::Value;
                em << it2->second.statSvd;

                em << YAML::Key;
                em << "avg";
                em << YAML::Value;
                em << it2->second.statAverage;

                em << YAML::Key;
                em << "histogram";
                em << YAML::Value;
                em << YAML::Flow << YAML::BeginSeq;
                for (std::size_t i = 0; i < it2->second.statHistogram.size(); ++i) {
                    em << it2->second.statHistogram[i];
                }
                em << YAML::EndSeq;
                em << YAML::EndMap;

#else
                em << YAML::Flow << YAML::BeginSeq;
                for (std::size_t i = 0; i < it2->second.results.size(); ++i) {
                    em << it2->second.results[i];
                }
                em << YAML::EndSeq;
#endif

            }
            em << YAML::EndMap;

            em << YAML::EndMap;
        }
        em << YAML::EndMap;
    }
    ofile << em.c_str();
    
    return 0;
} // renderImageSequence_main

int main(int argc, char** argv)
{
    try {
        return produceStatsMain(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s", e.what());
        exit(1);
    }
}
