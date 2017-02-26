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
#include <iostream>
#include <cmath>
#include <vector>
#include "DetectionsSerialization.h"
#include "SerializationIO.h"
#include "FStreamsSupport.h"

using std::string;

class RectD
{
public:

    double x1,y1,x2,y2;

    bool isNull() const
    {
        return (x2 <= x1) || (y2 <= y1);
    }

    double area() const
    {
        return (double)width() * height();
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

};

bool compareDetectionsAtFrame(int frameNumber,
                              int offset1,
                              int offset2,
                              const SERIALIZATION_NAMESPACE::SequenceSerialization& detectionSeq1,
                              const SERIALIZATION_NAMESPACE::SequenceSerialization& detectionSeq2,
                              const string& label,
                              double gtScaleX,
                              double gtScaleY,
                              bool* foundMatchingDetection,
                              RectD* gtRectOut,
                              RectD* detectionRectOut)
{
    const SERIALIZATION_NAMESPACE::DetectionSerialization* gtDetection = 0;
    {
        std::map<int, SERIALIZATION_NAMESPACE::FrameSerialization>::const_iterator foundFrameDet2 = detectionSeq2.frames.find(frameNumber + offset2);
        if (foundFrameDet2 != detectionSeq2.frames.end()) {
            for (std::list<SERIALIZATION_NAMESPACE::DetectionSerialization>::const_iterator it = foundFrameDet2->second.detections.begin() ; it != foundFrameDet2->second.detections.end(); ++it) {
                if (it->label == label) {
                    gtDetection = &(*foundFrameDet2->second.detections.begin());
                    break;
                }
            }
        }
    }
    if (!gtDetection) {
        std::cout << "No ground-truth data available at frame " << frameNumber << " , skipping this frame from the statistics" << std::endl;
        return false;
    }

    RectD gtRect;
    gtRect.x1 = gtDetection->x1 * gtScaleX;
    gtRect.y1 = gtDetection->y1 * gtScaleY;
    gtRect.x2 = gtDetection->x2 * gtScaleX;
    gtRect.y2 = gtDetection->y2 * gtScaleY;
    *gtRectOut = gtRect;

    double gtArea = gtRect.area();

    *foundMatchingDetection = false;
    {
        std::map<int, SERIALIZATION_NAMESPACE::FrameSerialization>::const_iterator foundFrameDet1 = detectionSeq1.frames.find(frameNumber + offset1);
        if (foundFrameDet1 == detectionSeq1.frames.end()) {
            return true;
        }

        RectD bestDetectionRect;
        double bestDetectionScore = 0;
        for (std::list<SERIALIZATION_NAMESPACE::DetectionSerialization>::const_iterator it = foundFrameDet1->second.detections.begin() ; it != foundFrameDet1->second.detections.end(); ++it) {
            RectD detectionRect;
            detectionRect.x1 = it->x1;
            detectionRect.y2 = it->y1;
            detectionRect.x2 = it->x2;
            detectionRect.y1 = it->y2;

            RectD intersection;
            if (!gtRect.intersect(detectionRect, &intersection)) {
                continue;
            }
            double intersectionArea = intersection.area();
            if (intersectionArea >= gtArea * 0.3) {

                if (bestDetectionRect.isNull()) {
                    bestDetectionRect = detectionRect;
                } else {
                    if (intersectionArea > bestDetectionScore) {
                        bestDetectionRect = detectionRect;
                        bestDetectionScore = intersectionArea;
                    }
                }
            }
        }
        if (!bestDetectionRect.isNull()) {
            *foundMatchingDetection = true;
            *detectionRectOut = bestDetectionRect;
        }
    }

    return true;


} // compareDetectionsAtFrame

extern image get_label(image **characters, char *string, int size);

void drawBbox(image& im, const RectD& bbox, const std::string& label, image **alphabet, int color_i)
{
    int width = im.h * .012;

    int offset = color_i * 123457 % 2;

    float red = get_color(2,offset,2);
    float green = get_color(1,offset,2);
    float blue = get_color(0,offset,2);
    float rgb[3];

    //width = prob*20+2;

    rgb[0] = red;
    rgb[1] = green;
    rgb[2] = blue;

    int left  = (int)bbox.x1;
    int right = (int)bbox.x2;
    int top   = (int)bbox.y2;
    int bot   = (int)bbox.y1;

    if(left < 0) left = 0;
    if(right > im.w-1) right = im.w-1;
    if(top < 0) top = 0;
    if(bot > im.h-1) bot = im.h-1;

    draw_box_width(im, left, top, right, bot, width, red, green, blue);
    if (alphabet) {
        image labelImg = get_label(alphabet, const_cast<char*>(label.c_str()), (im.h*.03)/10);
        draw_label(im, top + width, left, labelImg, rgb);
    }

}

void writeImage(const std::string& inputFilename,
                const std::string& outputFilename,
                const std::string& label,
                image **alphabet,
                const RectD& gtRect,
                const RectD& detectionRect)
{
    image inputImage = load_image_color(const_cast<char*>(inputFilename.c_str()),0,0);

    if (!gtRect.isNull()) {
        drawBbox(inputImage, gtRect, label + "Ground truth", alphabet, 0);
    }
    if (!detectionRect.isNull()) {
        drawBbox(inputImage, gtRect, label + "Detection", alphabet, 1);
    }
    save_image(inputImage, outputFilename.c_str());
}

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

static void parseArguments(const std::list<string>& args,
                           SequenceParsing::SequenceFromPattern* inputSequence,
                           int* firstFrame, int* lastFrame,
                           int* offset1, int *offset2,
                           double* gtScaleX, double* gtScaleY,
                           SERIALIZATION_NAMESPACE::SequenceSerialization* detectionFile1,
                           SERIALIZATION_NAMESPACE::SequenceSerialization* detectionFile2,
                           string* outputPattern,
                           string* label)
{
    std::list<string> localArgs = args;
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "-i");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("An input sequence file must be specified with the -i switch.");
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
        std::list<string>::iterator foundInput = hasToken(localArgs, "-d");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("2 detection files must be specified with the -d switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("-d switch without 2 file names");
        }
        {
            std::string detectionFileName1 = *foundInput;
            FStreamsSupport::ifstream ifile;
            FStreamsSupport::open(&ifile, detectionFileName1);
            if (!ifile) {
                throw std::invalid_argument("Could not open file " + detectionFileName1);
            }
            SERIALIZATION_NAMESPACE::read(SERIALIZATION_FILE_FORMAT_HEADER, ifile, detectionFile1);
        }

        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("-d switch without 2 file names");
        }
        {
            std::string detectionFileName2 = *foundInput;
            FStreamsSupport::ifstream ifile;
            FStreamsSupport::open(&ifile, detectionFileName2);
            if (!ifile) {
                throw std::invalid_argument("Could not open file " + detectionFileName2);
            }
            std::cout << "Assuming ground-truth detection file is " << detectionFileName2 << std::endl;
            SERIALIZATION_NAMESPACE::read(SERIALIZATION_FILE_FORMAT_HEADER, ifile, detectionFile2);
        }

        localArgs.erase(foundInput);
    }

    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--label");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("A label must be specified with the --label switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("--label switch without a label");
        } else {
            *label = *foundInput;
            localArgs.erase(foundInput);
        }
    }

    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--offset");
        *offset2 = *offset1 = 0;
        if (foundInput != localArgs.end()) {
            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--offset switch without 2 offsets");
            }
            {
                std::stringstream ss(*foundInput);
                ss >> *offset1;
            }
            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--offset switch without 2 offsets");
            }
            {
                std::stringstream ss(*foundInput);
                ss >> *offset2;
            }

        }

    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--gtScale");
        *gtScaleX = *gtScaleY = 1.;
        if (foundInput != localArgs.end()) {
            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--gtScale switch without X and Y scale");
            }
            {
                std::stringstream ss(*foundInput);
                ss >> *gtScaleX;
            }
            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--gtScale switch without X and Y scale");
            }
            {
                std::stringstream ss(*foundInput);
                ss >> *gtScaleY;
            }

        }
        
    }



    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "-o");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("An output file sequence pattern must be specified with the -o switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("-o switch without a file name");
        } else {
            *outputPattern = *foundInput;
            localArgs.erase(foundInput);
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
    int offset1, offset2;
    double gtScaleX, gtScaleY;
    SERIALIZATION_NAMESPACE::SequenceSerialization detection1, detection2;
    string outputPattern;
    string label;
    parseArguments(arguments, &inputFiles, &firstFrame, &lastFrame, &offset1, &offset2, &gtScaleX, &gtScaleY, &detection1, &detection2, &outputPattern, &label);

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
        throw std::invalid_argument("At least one frame must be provided");
    }

    image **alphabet = load_alphabet();

    int curFrame_i = 0;

    int nbMatches = 0;

    while (curFrame_i < (int)inputFilesInOrder.size()) {

        int frameNumber = inputFilesInOrder[curFrame_i].second;
        const std::string& filename = inputFilesInOrder[curFrame_i].first;
        std::string outputFilename = SequenceParsing::generateFileNameFromPattern(outputPattern, std::vector<std::string>(), curFrame_i, 0);

        RectD gtRect, detectionRect;
        bool foundMatch = false;
        bool hasGroundTruth = compareDetectionsAtFrame(frameNumber, offset1, offset2, detection1, detection2, label, gtScaleX, gtScaleY, &foundMatch, &gtRect, &detectionRect);

        if (foundMatch) {
            std::cout << "Found matching detection for frame " << frameNumber << std::endl;
        } else {
            std::cout << "Could not find a matching detection for frame " << frameNumber << std::endl;
        }
        writeImage(filename, outputFilename, label, alphabet, gtRect, detectionRect);

        if (hasGroundTruth) {
            if (foundMatch) {
                ++nbMatches;
            }
        }
        ++curFrame_i;
    }

    std::cout << "Total matched %: " << nbMatches / inputFilesInOrder.size() << std::endl;


    return 0;
} // renderImageSequence_main

int main(int argc, char** argv)
{
    try {
        return renderImageSequence_main(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s", e.what());
        exit(1);
    }
}
