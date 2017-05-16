#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include <sys/time.h>

#include "SequenceParsing.h"
#include <list>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>
#include "DetectionsSerialization.h"
#include "SerializationIO.h"
#include "FStreamsSupport.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#endif

using std::string;

class RectI
{
public:

    int x1; // left
    int y1; // bottom
    int x2; // right
    int y2; // top

    RectI()
    : x1(0), y1(0), x2(0), y2(0)
    {
    }

    int width() const
    {
        return x2 - x1;
    }

    int height() const
    {
        return y2 - y1;
    }

    /// returns true if the rect passed as parameter  intersects this one
    bool intersects(const RectI & r) const
    {
        if ( isNull() || r.isNull() ) {
            return false;
        }
        if ( (r.x2 <= x1) || (x2 <= r.x1) || (r.y2 <= y1) || (y2 <= r.y1) ) {
            return false;
        }

        return true;
    }

    bool intersect(const RectI & r,
                   RectI* intersection) const
    {
        if ( !intersects(r) ) {
            return false;
        }

        intersection->x1 = std::max(x1, r.x1);
        intersection->y1 = std::max(y1, r.y1);
        // the region must be *at least* empty, thus the maximin.
        intersection->x2 = std::max( intersection->x1, std::min(x2, r.x2) );
        // the region must be *at least* empty, thus the maximin.
        intersection->y2 = std::max( intersection->y1, std::min(y2, r.y2) );

        assert( !intersection->isNull() );

        return true;
    }


    bool isNull() const
    {
        return (x2 <= x1) || (y2 <= y1);
    }

    bool contains(const RectI & other) const
    {
        return other.x1 >= x1 &&
        other.y1 >= y1 &&
        other.x2 <= x2 &&
        other.y2 <= y2;
    }
};

class RectD
{
public:

    double x1; // left
    double y1; // bottom
    double x2; // right
    double y2; // top

    RectD()
    : x1(0), y1(0), x2(0), y2(0)
    {
    }


    double width() const
    {
        return x2 - x1;
    }

    double height() const
    {
        return y2 - y1;
    }

    bool intersect(const RectD & r,
                   RectD* intersection) const
    {
        if ( isNull() || r.isNull() ) {
            return false;
        }

        if ( (x1 > r.x2) || (r.x1 > x2) || (y1 > r.y2) || (r.y1 > y2) ) {
            return false;
        }

        intersection->x1 = std::max(x1, r.x1);
        intersection->x2 = std::min(x2, r.x2);
        intersection->y1 = std::max(y1, r.y1);
        intersection->y2 = std::min(y2, r.y2);

        return true;
    }

    bool isNull() const
    {
        return (x2 <= x1) || (y2 <= y1);
    }
};

struct FetcherThreadArgs
{
    image inputImage, inputImageScaled;
    string filename;
    network* net;
};

void *fetch_in_thread(void *ptr)
{
    FetcherThreadArgs* args = (FetcherThreadArgs*)ptr;
    args->inputImage = load_image_color(const_cast<char*>(args->filename.c_str()),0,0);
    args->inputImageScaled = resize_image(args->inputImage, args->net->w, args->net->h);
    return 0;
}

struct DetectThreadArgs
{
    // Changes for each detection
    int frameNumber;
    image inputImage, inputImageScaled;

    // Return code
    bool ok;

    // Never changes
    network* net;
    float* predictions;
    float** probs;
    float* avg;
    box *boxes;
    float thresh;
    float hierThresh;
    std::vector<string>* names;
    std::vector<string>* allowedClasses;
    FStreamsSupport::ofstream* modelFile;
    string *outputFilename;
    SERIALIZATION_NAMESPACE::SequenceSerialization* outSeq;

};

#ifdef OPENCV
/**
 * @brief Computes H and S histograms for the given rectangle
 **/
static void computeHistograms(const RectI& roi, const image& image, SERIALIZATION_NAMESPACE::DetectionSerialization* serialization)
{
    RectI imageBounds;
    imageBounds.x1 = 0;
    imageBounds.y1 = 0;
    imageBounds.x2 = image.w;
    imageBounds.y2 = image.h;
    assert(imageBounds.contains(roi));



    // Convert the image to a cv::Mat
    cv::Mat rgbMat(roi.height(), roi.width(), CV_8UC3);
    cv::Mat hsvMat;

    {
        uchar* dstPixels = hsvMat.data;
        assert(hsvMat.step.p[1] == 3);

        float* srcPixels = image.data + roi.y1 * image.w * image.c + roi.x1 * image.c;

        //return data + i0 * step.p[0] + i1 * step.p[1];
        for (int y = roi.y1; y < roi.y2; ++y) {
            for (int x = roi.x1; x < roi.x2; ++x,dstPixels += 3, srcPixels += image.c) {
                for (int c = 0; c < 3; ++c) {
                    float srcVal;
                    if (c < image.c) {
                        srcVal = srcPixels[c];
                    } else {
                        srcVal = 0;
                    }
                    dstPixels[c] = std::min(1.f,std::max(0.f, srcVal)) / 255.f;
                }
            }
            // Remove what was done on last iteration and go to the next line
            srcPixels += (image.w * image.c - roi.width() * image.c);
            dstPixels += (hsvMat.step.p[0] - roi.width() * 3);
        }
    }

    // Convert to HSV
    cv::cvtColor( rgbMat, hsvMat, cv::COLOR_BGR2HSV);

    // Using 32 bins for hue and 32 for saturation
    int histSize[] = { 32, 32 };

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };

    // Use the 0-th and 1-st channels
    int channels[] = { 0, 1 };

    cv::Mat hist;

    // Calculate the histograms for the HSV images
    calcHist( &hsvMat, 1, channels, cv::Mat() /*mask*/, hist, 2, histSize, ranges, true /*uniform*/, false /*accumulate*/);

    // normalize between 0 and 1
    cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    serialization->histogramSizes.resize(2);
    serialization->histogramSizes[0] = histSize[0];
    serialization->histogramSizes[1] = histSize[1];

    serialization->histogramData.resize(hist.rows * hist.cols);

    double* histData = &serialization->histogramData[0];
    for (int i = 0; i < hist.rows; ++i) {
        for (int j = 0; j < hist.cols; ++j) {
            *histData = hist.at<double>(i, j);
            ++histData;
        }
    }

    // Write in output
}
#endif // #ifdef OPENCV

void *detect_in_thread(void *ptr)
{

    DetectThreadArgs* args = (DetectThreadArgs*)ptr;
    args->ok = true;

    float nms = .4;

    layer l = args->net->layers[args->net->n-1];
    float *X = args->inputImageScaled.data;
    float *prediction = network_predict(*args->net, X);

    memcpy(args->predictions, prediction, l.outputs*sizeof(float));

    mean_arrays(&args->predictions, 1, l.outputs, args->avg);
    l.output = args->avg;

    free_image(args->inputImageScaled);

    if (l.type == DETECTION) {
        get_detection_boxes(l, 1, 1, args->thresh, args->probs, args->boxes, 0);
    } else if (l.type == REGION) {
        get_region_boxes(l, 1, 1, args->thresh, args->probs, args->boxes, 0, 0, args->hierThresh);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) {
        do_nms(args->boxes, args->probs, l.w*l.h*l.n, l.classes, nms);
    }

    int gridSize = l.w * l.h * l.n;
    int nClasses = (int)args->names->size();

    int imgWidth = args->inputImage.w;
    int imgHeight = args->inputImage.h;


    SERIALIZATION_NAMESPACE::FrameSerialization frameSerialization;

    for (int i = 0; i < gridSize; ++i) {
        int class1 = max_index(args->probs[i], nClasses);
        float prob = args->probs[i][class1];
        if (prob <= args->thresh) {
            continue;
        }

        std::string& label = (*args->names)[class1];

        bool keepDetection = false;
        for (std::size_t c = 0; c < args->allowedClasses->size(); ++c) {
            if ((*args->allowedClasses)[c] == label) {
                keepDetection = true;
                break;
            }
        }
        if (!keepDetection) {
            continue;
        }

        SERIALIZATION_NAMESPACE::DetectionSerialization detection;
        detection.label = label;
        detection.score = std::max(0.f,std::min(1.f, prob));


        const box& b = args->boxes[i];

        detection.x1 = (b.x-b.w/2.) * imgWidth;
        detection.y1 = (b.y+b.h/2.) * imgHeight;
        detection.y1 = imgHeight - 1 - detection.y1;
        detection.x2 = (b.x+b.w/2.) * imgWidth + 1;
        detection.y2 = (b.y-b.h/2.) * imgHeight;
        detection.y2 = imgHeight - detection.y2;

        // Clamp
        detection.x1 = std::max(0.,std::min((double)imgWidth, detection.x1));
        detection.y1 = std::max(0.,std::min((double)imgHeight, detection.y1));
        detection.x2 = std::max(0.,std::min((double)imgWidth, detection.x2));
        detection.y2 = std::max(0.,std::min((double)imgHeight, detection.y2));
        
#ifdef OPENCV
        RectI roi;
        roi.x1 = std::floor(detection.x1);
        roi.y1 = std::floor(detection.y1);
        roi.x2 = std::ceil(detection.x2);
        roi.y2 = std::ceil(detection.y2);
        computeHistograms(roi, args->inputImage, &detection);
#endif

        frameSerialization.detections.push_back(detection);

    }

    free_image(args->inputImage);


    // No detection, do not write to the file
    if (frameSerialization.detections.empty()) {
        return 0;
    }

    args->outSeq->frames.insert(std::make_pair(args->frameNumber, frameSerialization));

    // Write the results file for each frame in case it crash so nothing is lost
    FStreamsSupport::ofstream ofile;
    FStreamsSupport::open(&ofile, *args->outputFilename);
    if (!ofile) {
        args->ok = false;
        return 0;
    }
    try {
        SERIALIZATION_NAMESPACE::write(ofile, *args->outSeq, SERIALIZATION_FILE_FORMAT_HEADER);
    } catch (const std::exception& e) {
        args->ok = false;
        fprintf(stderr, "%s", e.what());
        return 0;
    }


    return 0;
} // detect_in_thread

static std::list<string>::iterator hasToken(std::list<string> &localArgs, const string& token)
{
    for (std::list<string>::iterator it = localArgs.begin(); it!=localArgs.end(); ++it) {
        if (*it == token) {
            return it;
        }
    }
    return localArgs.end();
} // hasToken

void split(const std::string &s, char delim, std::vector<std::string>* result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result->push_back(item);
    }
}

static void printUsage(const char* argv0)
{
    /* Text must hold in 80 columns ************************************************/
    std::stringstream ss;
    ss << "Usage:\n"
    << argv0
    <<
    " -i <inputVideo> -o <outputDetectionFile> --cfg <yoloConfigFile> --weights <yoloWeightsFile> [--names <classNamesFile>] [--range <firstFrame-lastFrame>] [--threshold <value>]\n"
    ;
    std::cout << ss.str() << std::endl;

}

static void parseArguments(const std::list<string>& args,
                           SequenceParsing::SequenceFromPattern* inputSequence,
                           int* firstFrame, int* lastFrame,
                           string* cfgFilename,
                           string* weightFilename,
                           float* thresh,
                           std::vector<string>* names,
                           string* outputFilename)
{
    std::list<string> localArgs = args;
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "-i");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("An input file must be specified with the -i switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("-i switch without a file name");
        } else {
            SequenceParsing::filesListFromPattern_slow(*foundInput, inputSequence);
            localArgs.erase(foundInput);
        }
    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--range");
        if (foundInput == localArgs.end()) {
            *firstFrame = inputSequence->begin()->first;
            *lastFrame = inputSequence->rbegin()->first;
        } else {
            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--range switch without a frame range");
            } else {
                std::vector<string> rangeVec;
                split(*foundInput, '-', &rangeVec);
                if (rangeVec.size() != 2) {
                    throw std::invalid_argument("Frame range must be in the form firstFrame-lastFrame");
                }
                {
                    std::stringstream ss(rangeVec[0]);
                    ss >> *firstFrame;
                }
                {
                    std::stringstream ss(rangeVec[1]);
                    ss >> *lastFrame;
                }
                if (*firstFrame < inputSequence->begin()->first || *lastFrame > inputSequence->rbegin()->first) {
                    printf("Frame range must be in the sequence range (%i-%i)",inputSequence->begin()->first, inputSequence->rbegin()->first);
                    exit(1);
                }
                localArgs.erase(foundInput);
            }
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
            *outputFilename = *foundInput;
            localArgs.erase(foundInput);
        }
    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--cfg");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("A config file must be specified with the --cfg switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("--cfg switch without a file name");
        } else {
            *cfgFilename = *foundInput;
            localArgs.erase(foundInput);
        }
    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--weights");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("A config file must be specified with the --weights switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("--weights switch without a file name");
        } else {
            *weightFilename = *foundInput;
            localArgs.erase(foundInput);
        }
    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--names");
        std::string namesFile;
        if (foundInput == localArgs.end()) {
            printf("--names switch not specified, using data/names.list instead");
            namesFile = "data/names.list";
        } else {
            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--names switch without a file name");
            } else {
                namesFile = *foundInput;
                localArgs.erase(foundInput);
            }
        }
        {
            std::ifstream ifile;
            ifile.open(namesFile);
            if (!ifile) {
                fprintf(stderr, "Couldn't open file: %s\n", namesFile.c_str());
                exit(1);
            }
            string lineBuf;
            while (std::getline(ifile, lineBuf)){
                names->push_back(lineBuf);
            }
        }

    }

    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--thresh");
        if (foundInput == localArgs.end()) {
            *thresh = 0.25;
            printf("--thresh not specified, using %f", *thresh);
        } else {
            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--thresh switch without a value");
            } else {
                std::stringstream ss(*foundInput);
                ss >> *thresh;
                localArgs.erase(foundInput);
            }
        }
    }

} // parseArguments



int renderImageSequence_main(int argc, char** argv)
{
    std::list<string> arguments;
    for (int i = 0; i < argc; ++i) {
        arguments.push_back(string(argv[i]));
    }

    SequenceParsing::SequenceFromPattern inputFiles;
    int firstFrame, lastFrame;
    string cfgFilename, weightFilename;
    float thresh;
    std::vector<string> names;
    string outputFile;
    float hier_thresh = .5;
    std::vector<string> allowedNames;
    allowedNames.push_back("person");
    parseArguments(arguments, &inputFiles, &firstFrame, &lastFrame, &cfgFilename, &weightFilename, &thresh, &names, &outputFile);

    // vector of pair<filename, framenumber> >
    std::vector<std::pair<std::string,int> > inputFilesInOrder;

    {
        SequenceParsing::SequenceFromPattern::iterator startIt = inputFiles.find(firstFrame);
        SequenceParsing::SequenceFromPattern::iterator endIt = inputFiles.find(lastFrame);
        if (startIt == inputFiles.end() || endIt == inputFiles.end()) {
            throw std::invalid_argument("Invalid frame range");
        }
        ++endIt;
        for (SequenceParsing::SequenceFromPattern::iterator it = startIt; it != endIt; ++it) {
            inputFilesInOrder.push_back(std::make_pair(it->second.begin()->second, it->first));
        }
    }
    if (inputFilesInOrder.empty()) {
        throw std::invalid_argument("At least one frame must be detected");
    }

    SERIALIZATION_NAMESPACE::SequenceSerialization detectionResults;

    network net = parse_network_cfg(const_cast<char*>(cfgFilename.c_str()));
    load_weights(&net, const_cast<char*>(weightFilename.c_str()));
    set_batch_network(&net, 1);
    
    srand(2222222);
    
    layer detectionLayer = net.layers[net.n-1];
    
    float* predictions = (float *) calloc(detectionLayer.outputs, sizeof(float));
    float* avg = (float *) calloc(detectionLayer.outputs, sizeof(float));

    int gridSize = detectionLayer.w * detectionLayer.h * detectionLayer.n;

    box* boxes = (box *)calloc(gridSize, sizeof(box));
    float** probs = (float **)calloc(gridSize, sizeof(float *));

    for (int j = 0; j < gridSize; ++j) {
        probs[j] = (float *)calloc(detectionLayer.classes, sizeof(float));
    }


    // Name the model file containing the histograms exactly like outputFile but ending with _model
    std::string modelFileName = outputFile;
    {
        std::size_t foundLastDot = modelFileName.find_last_of(".");
        if (foundLastDot != std::string::npos) {
            modelFileName = modelFileName.substr(0, foundLastDot);
        }
        modelFileName += "_model";
    }

    FStreamsSupport::ofstream modelFile;
    FStreamsSupport::open(&modelFile, modelFileName, std::ios_base::out | std::ios_base::trunc);

    int curFrame_i = 0;

    FetcherThreadArgs fetchArgs;
    fetchArgs.net = &net;

    DetectThreadArgs detectArgs;
    detectArgs.net = &net;
    detectArgs.predictions = predictions;
    detectArgs.probs = probs;
    detectArgs.avg = avg;
    detectArgs.boxes = boxes;
    detectArgs.thresh = thresh;
    detectArgs.hierThresh = hier_thresh;
    detectArgs.names = &names;
    detectArgs.modelFile = &modelFile;
    detectArgs.outSeq = &detectionResults;
    detectArgs.outputFilename = &outputFile;
    detectArgs.allowedClasses = &allowedNames;

    while (curFrame_i < (int)inputFilesInOrder.size()) {

        int frameNumber = inputFilesInOrder[curFrame_i].second;
        const std::string& filename = inputFilesInOrder[curFrame_i].first;

        fetchArgs.filename = filename;
        fetch_in_thread(&fetchArgs);

        detectArgs.frameNumber = frameNumber;
        detectArgs.inputImage = fetchArgs.inputImage;
        detectArgs.inputImageScaled = fetchArgs.inputImageScaled;
        std::cout << "Running detector on frame " << frameNumber << std::endl;
        detect_in_thread(&detectArgs);
        if (!detectArgs.ok) {
            return 1;
        }

        ++curFrame_i;
    }
    return 0;
} // renderImageSequence_main

int main(int argc, char** argv)
{
    try {
        return renderImageSequence_main(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s", e.what());
        printUsage(argv[0]);
        exit(1);
    }
}
