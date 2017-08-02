#include "detectorCommon.h"
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string.hpp>

#include <yaml-cpp/yaml.h>

#define kDwKey "dw"
#define kDhKey "dh"
#define kDxKey "dx"
#define kDyKey "dy"
#define kDHistChi2Key "dhist_chi2"
#define kDModelChi2Key "dModel_chi2"
#define kTransitionCostKey "transitionCost"

#define kSubTransitionDx "transition_dx"
#define kSubTransitionDy "transition_dy"
#define kSubTransitionDw "transition_dw"
#define kSubTransitionDh "transition_dh"
#define kSubTransitionDHistChi2 "transition_dHistChi2"
#define kSubTransitionDModelChi2 "transition_dModelChi2"

#define kCategoryNameSameActor "SameActor"
#define kCategoryDifferentActors "DifferentActors"

using std::string;

struct SampleData
{
    // Raw samples (dx, dy, dw, dh, histogram distance etc...) identified with a key defined above
    std::map<std::string, double> values;
};

struct ActorToActorKey
{
    std::string category;
    int frameOffset;
};

struct ActorToActorKeyCompareLess
{
    bool operator()(const ActorToActorKey& lhs, const ActorToActorKey& rhs) const
    {
        int c = lhs.category.compare(rhs.category);
        if (c < 0) {
            return true;
        } else if (c > 0) {
            return false;
        }
        return lhs.frameOffset < rhs.frameOffset;
    }
};


// Statistic over a specific sample variable (e.g: dx)
struct StatResults
{
    double statAverage;
    double statMin, statMax;
    double statStdev;

    // All samples concatenated in a single array
    std::vector<double> results;


    StatResults()
    : statAverage(0)
    , statMin(std::numeric_limits<double>::infinity())
    , statMax(-std::numeric_limits<double>::infinity())
    , statStdev(0)
    , results()
    {
        
    }
};

typedef std::map<string, StatResults> StatResultsMap;


struct ActorToActorData
{
    // For each sample, we have many variables
    std::list<SampleData> samples;

    // Statistics for each variable
    StatResultsMap results;

};

typedef std::map<ActorToActorKey, ActorToActorData, ActorToActorKeyCompareLess> PerActorStatDataMap;

static void parseArguments(const std::list<string>& args,
                           string *detectionsFile,
                           string *outputFileName,
                           int* width,
                           int* height,
                           std::map<std::string,int>* actorsModelFrame)
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
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "-f");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("Image format (e.g: 1920x1080) must be specified with the -f switch");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("-f switch without a format");
        } else {
            std::string formatStr = *foundInput;
            std::vector<string> splits;
            boost::split(splits, formatStr,boost::is_any_of("x"));
            if (splits.size() != 2) {
                throw std::invalid_argument("Format must be in the form \"widthxheight\"");
            }
            for (std::size_t i = 0; i < splits.size(); ++i) {
                // Remove whitespaces
                boost::trim(splits[i]);
            }

            *width = boost::lexical_cast<int>(splits[0]);
            *height = boost::lexical_cast<int>(splits[1]);

            localArgs.erase(foundInput);
        }
    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "-m");
        while (foundInput != localArgs.end()) {

            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("-m switch without an actor name");
            }

            std::string actorName = *foundInput;

            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("-m switch without an actor model frame");
            }

            int actorModelFrame = boost::lexical_cast<int>(*foundInput);

            actorsModelFrame->insert(std::make_pair(actorName, actorModelFrame));

            foundInput = hasToken(localArgs, "-m");
        }

    }
} // parseArguments

static void compareDetections(const SERIALIZATION_NAMESPACE::DetectionSerialization& a, const SERIALIZATION_NAMESPACE::DetectionSerialization& b,
                              const std::vector<boost::shared_ptr<FStreamsSupport::ifstream> >& modelFiles,
                              const std::vector<double>& histogramA,
                              const std::vector<double>& histogramB,
                              int width, int height,
                              SampleData* data)
{
    RectD aRect(a.x1, a.y1, a.x2, a.y2);
    RectD bRect(b.x1, b.y1, b.x2, b.y2);

    data->values[kDwKey] = std::abs(aRect.width() - bRect.width()) / (double)width;
    data->values[kDhKey] = std::abs(aRect.height() - bRect.height()) / (double)height;
    data->values[kDxKey] = std::abs((aRect.x1 + aRect.x2) / 2. -  (bRect.x1 + bRect.x2) / 2.) / (double)width;
    data->values[kDyKey] = std::abs((aRect.y1 + aRect.y2) / 2. -  (bRect.y1 + bRect.y2) / 2.) / (double)height;
    data->values[kDHistChi2Key] = compareHist(histogramA, histogramB, eHistogramComparisonChiSquare);
}

struct ActorModel
{
    std::vector<double> histogram;
};

typedef std::map<std::string, ActorModel> PerActorModel;

static void computeData(const string& inputFilename, const string& outputFilename, const std::vector<int>& frameOffsets, int width, int height, const std::map<std::string,int>& actorsModelFrame, PerActorStatDataMap& data) {

    assert(!frameOffsets.empty());

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

    PerActorModel actorsModel;

    {
        for (std::map<int, SERIALIZATION_NAMESPACE::FrameSerialization>::const_iterator it = detectionResults.frames.begin(); it != detectionResults.frames.end(); ++it) {

            for (std::list<SERIALIZATION_NAMESPACE::DetectionSerialization>::const_iterator it2 = it->second.detections.begin(); it2 != it->second.detections.end(); ++it2) {

                std::vector<double> histogramA;
                assert(it2->fileIndex < (int)modelFiles.size());
                loadHistogramFromFile(modelFiles[it2->fileIndex], it2->modelIndex, detectionResults.histogramSizes, &histogramA);

                ActorModel& model = actorsModel[it2->label];
                if (model.histogram.empty()) {
                    // Compute model at frame specified by user, otherwise use the current frame to initialize
                    std::map<std::string,int>::const_iterator foundActorRefFrame = actorsModelFrame.find(it2->label);
                    if (foundActorRefFrame == actorsModelFrame.end()) {
                        model.histogram = histogramA;
                    } else {

                        // Look for the actor data at the specified frame
                        std::map<int, SERIALIZATION_NAMESPACE::FrameSerialization>::const_iterator foundFrameSerialization = detectionResults.frames.find(foundActorRefFrame->second);
                        if (foundFrameSerialization != detectionResults.frames.end()) {
                            for (std::list<SERIALIZATION_NAMESPACE::DetectionSerialization>::const_iterator it3 = foundFrameSerialization->second.detections.begin(); it3 != foundFrameSerialization->second.detections.end(); ++it3) {
                                if (it3->label == it2->label) {
                                    // We found the actor
                                    loadHistogramFromFile(modelFiles[it3->fileIndex], it3->modelIndex, detectionResults.histogramSizes, &model.histogram);
                                    break;
                                }
                            }
                        }
                        if (model.histogram.empty()) {
                            model.histogram = histogramA;
                        }
                    }
                }



                for (std::vector<int>::const_iterator itoff = frameOffsets.begin(); itoff != frameOffsets.end(); ++itoff) {
                    std::map<int, SERIALIZATION_NAMESPACE::FrameSerialization>::const_iterator next = it;
                    bool gotNext = true;
                    for (int i = 0; i < *itoff; ++i) {
                        if (next != detectionResults.frames.end()) {
                            ++next;
                        } else {
                            gotNext = false;
                            break;
                        }
                    }

                    if (!gotNext || next == detectionResults.frames.end()) {
                        continue;
                    }


                    assert(next->first > it->first);

                    // Search the same actor in the next frame
                    for (std::list<SERIALIZATION_NAMESPACE::DetectionSerialization>::const_iterator it3 = next->second.detections.begin(); it3 != next->second.detections.end(); ++it3) {
                        ActorToActorKey key;

                        bool sameActor = it2->label == it3->label;
                        key.category = sameActor ? kCategoryNameSameActor : kCategoryDifferentActors;
                        key.frameOffset = next->first - it->first;

                        // Only compare detections with a step of more than 1 frame for the same actor: it allows to have statistics for a hidden actor.
                        if (!sameActor && key.frameOffset > 1) {
                            continue;
                        }

                        std::vector<double> histogramB;
                        assert(it3->fileIndex < (int)modelFiles.size());
                        loadHistogramFromFile(modelFiles[it3->fileIndex], it3->modelIndex, detectionResults.histogramSizes, &histogramB);



                        ActorToActorData& actorsData = data[key];

                        SampleData statistics;
                        double distToModel = compareHist(histogramB, model.histogram, eHistogramComparisonChiSquare);
                        statistics.values[kDModelChi2Key] = distToModel;
                        compareDetections(*it2, *it3, modelFiles, histogramA, histogramB, width, height, &statistics);
                        actorsData.samples.push_back(statistics);
                    }

                }

            }
            
        }
    }
}


inline double mahalanobisValue(double dx, double sdev)
{
    assert(sdev > 0);
    //return (dx * dx) / (sdev * sdev);
    return std::abs(dx) / sdev;
}

inline double mahalanobisDistance(double dx, double dy, double dw, double dh, double sdevX, double sdevY, double sdevW, double sdevH)
{
    double xS = mahalanobisValue(dx, sdevX);
    double yS = mahalanobisValue(dy, sdevY);
    double wS = mahalanobisValue(dw, sdevW);
    double hS = mahalanobisValue(dh, sdevH);
    //double ret = std::sqrt(xS + yS + wS + hS);
    double ret = xS + yS + wS + hS;
    return ret;
}

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
    std::map<std::string,int> actorsModelFrame;
    int width, height;
    parseArguments(arguments, &inputFilename, &outputFilename, &width, &height, &actorsModelFrame);

    FStreamsSupport::ofstream ofile;
    FStreamsSupport::open(&ofile, outputFilename, std::ios_base::out | std::ios_base::trunc);
    if (!ofile) {
        throw std::invalid_argument("Could not open " + outputFilename);
    }

    std::vector<int> frameOffsets;
    frameOffsets.push_back(1);
    frameOffsets.push_back(24);
    frameOffsets.push_back(48);
    frameOffsets.push_back(120);

    PerActorStatDataMap data;
    computeData(inputFilename, outputFilename, frameOffsets, width, height, actorsModelFrame, data);


    for (PerActorStatDataMap::iterator it = data.begin(); it != data.end(); ++it) {

        if (it->second.samples.size() == 0) {
            continue;
        }

        // Compute average, min, max
        for (std::list<SampleData>::const_iterator it2 = it->second.samples.begin(); it2 != it->second.samples.end(); ++it2) {

            for (std::map<std::string, double>::const_iterator it3 = it2->values.begin(); it3 != it2->values.end(); ++it3) {
                StatResults& statValue = it->second.results[it3->first];
                statValue.statAverage += it3->second;
                statValue.statMin = std::min(statValue.statMin, it3->second);
                statValue.statMax = std::max(statValue.statMax, it3->second);
                statValue.results.push_back(it3->second);
            }
        }
    }

    ActorToActorData* sameActorFrameOffset1Data = 0;
    for (PerActorStatDataMap::iterator it = data.begin(); it != data.end(); ++it) {
        if (it->first.category == kCategoryNameSameActor && it->first.frameOffset == 1) {
            sameActorFrameOffset1Data = &it->second;
        }
        if (it->second.samples.size() == 0) {
            continue;
        }
        for (StatResultsMap::iterator it2 = it->second.results.begin(); it2 != it->second.results.end(); ++it2) {

            StatResults &stat = it2->second;
            stat.statAverage /= it->second.samples.size();

            for (std::vector<double>::iterator it3 = stat.results.begin(); it3 != stat.results.end(); ++it3) {
                stat.statStdev += (*it3 - stat.statAverage) * (*it3 - stat.statAverage);

            }
            stat.statStdev = std::sqrt(stat.statStdev / it->second.samples.size());
        }

    }
    assert(sameActorFrameOffset1Data);

    double sdevX = sameActorFrameOffset1Data->results[kDxKey].statStdev * width;
    double sdevY = sameActorFrameOffset1Data->results[kDyKey].statStdev * height;
    double sdevW = sameActorFrameOffset1Data->results[kDwKey].statStdev * width;
    double sdevH = sameActorFrameOffset1Data->results[kDhKey].statStdev * height;
    double sdevHisto = sameActorFrameOffset1Data->results[kDHistChi2Key].statStdev;
    double sdevModel = sameActorFrameOffset1Data->results[kDModelChi2Key].statStdev;

    std::vector<std::string> statsToCompute;
    statsToCompute.push_back(kSubTransitionDx);
    statsToCompute.push_back(kSubTransitionDy);
    statsToCompute.push_back(kSubTransitionDw);
    statsToCompute.push_back(kSubTransitionDh);
    statsToCompute.push_back(kSubTransitionDHistChi2);
    statsToCompute.push_back(kSubTransitionDModelChi2);
    statsToCompute.push_back(kTransitionCostKey);

    for (PerActorStatDataMap::iterator it = data.begin(); it != data.end(); ++it) {


        // Compute transition costs

        for (std::list<SampleData>::iterator it2 = it->second.samples.begin(); it2 != it->second.samples.end(); ++it2) {

            double dx = it2->values[kDxKey] * width;
            double dy = it2->values[kDyKey] * height;
            double dw = it2->values[kDwKey] * width;
            double dh = it2->values[kDhKey] * height;
            double dhistChi2 = it2->values[kDHistChi2Key];
            double dModelChi2 = it2->values[kDModelChi2Key];

            double xS = mahalanobisValue(dx, sdevX);
            double yS = mahalanobisValue(dy, sdevY);
            double wS = mahalanobisValue(dw, sdevW);
            double hS = mahalanobisValue(dh, sdevH);

            double cost = xS + yS + wS + hS + dhistChi2 / sdevHisto + dModelChi2 / sdevModel;
            if (cost >= 56) {
                assert(true);
            }

            it2->values[kTransitionCostKey] = cost;
            it2->values[kSubTransitionDx] = xS;
            it2->values[kSubTransitionDy] = yS;
            it2->values[kSubTransitionDw] = wS;
            it2->values[kSubTransitionDh] = hS;
            it2->values[kSubTransitionDHistChi2] = dhistChi2 / sdevHisto;
            it2->values[kSubTransitionDModelChi2] = dModelChi2 / sdevModel;

            for (std::vector<std::string>::const_iterator it3 = statsToCompute.begin(); it3 != statsToCompute.end(); ++it3) {
                StatResults& costStats = it->second.results[*it3];
                costStats.results.push_back(it2->values[*it3]);
                costStats.statAverage += costStats.results.back();
                costStats.statMin = std::min(costStats.statMin, costStats.results.back());
                costStats.statMax = std::max(costStats.statMax, costStats.results.back());
            }
        }

    }

    for (PerActorStatDataMap::iterator it = data.begin(); it != data.end(); ++it) {

        if (it->second.samples.size() == 0) {
            continue;
        }

        for (std::vector<std::string>::const_iterator itStat = statsToCompute.begin(); itStat != statsToCompute.end(); ++itStat) {
            StatResults& transitionCostStats = it->second.results[*itStat];

            transitionCostStats.statAverage /= it->second.samples.size();

            for (std::vector<double>::iterator it3 = transitionCostStats.results.begin(); it3 != transitionCostStats.results.end(); ++it3) {
                transitionCostStats.statStdev += (*it3 - transitionCostStats.statAverage) * (*it3 - transitionCostStats.statAverage);

            }
            transitionCostStats.statStdev = std::sqrt(transitionCostStats.statStdev / it->second.samples.size());

        }

    }

    YAML::Emitter em;
    {
        std::cout << "==============Dumping stats==============" << std::endl << std::endl;

        em << YAML::BeginMap;
        for (PerActorStatDataMap::iterator it = data.begin(); it != data.end(); ++it) {

            std::string categoryLabel;
            if (it->first.category != kCategoryNameSameActor) {
                categoryLabel = it->first.category;
            } else {
                std::stringstream ss;
                ss << it->first.category;
                ss << "_offset=" << it->first.frameOffset;
                categoryLabel = ss.str();
            }
            std::cout << "============================" << std::endl << std::endl;
            std::cout << "Statistics for category " << categoryLabel << ":" << std::endl << std::endl;

            em << YAML::Key << categoryLabel;
            em << YAML::Value;
            const ActorToActorData& resultsData = it->second;
            em << YAML::BeginMap;
            em << YAML::Key << "NSamples" <<  YAML::Value << resultsData.samples.size();

            em << YAML::Key << "Stats" << YAML::Value;
            em << YAML::BeginMap;
            for (StatResultsMap::const_iterator it2 = resultsData.results.begin(); it2 != resultsData.results.end(); ++it2) {

                std::cout << it2->first << ": ";

                em << YAML::Key << it2->first;
                em << YAML::Value;

                em << YAML::BeginMap;
                em << YAML::Key;
                em << "min";
                em << YAML::Value;
                em << it2->second.statMin;
                std::cout << "min = " << it2->second.statMin;

                em << YAML::Key;
                em << "max";
                em << YAML::Value;
                em << it2->second.statMax;
                std::cout << ", max = " << it2->second.statMax;

                em << YAML::Key;
                em << "stdev";
                em << YAML::Value;
                em << it2->second.statStdev;
                std::cout << ", stdev = " << it2->second.statStdev;

                em << YAML::Key;
                em << "avg";
                em << YAML::Value;
                em << it2->second.statAverage;
                std::cout << ", avg = " << it2->second.statAverage << std::endl;

                /*em << YAML::Key;
                em << "histogram";
                em << YAML::Value;
                em << YAML::Flow << YAML::BeginSeq;
                for (std::size_t i = 0; i < it2->second.statHistogram.size(); ++i) {
                    em << it2->second.statHistogram[i];
                }
                em << YAML::EndSeq;*/

                em << YAML::Key;
                em << "values";
                em << YAML::Value;
                em << YAML::Flow << YAML::BeginSeq;
                for (std::size_t i = 0; i < it2->second.results.size(); ++i) {
                    em << it2->second.results[i];
                }
                em << YAML::EndSeq;

                em << YAML::EndMap;


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
