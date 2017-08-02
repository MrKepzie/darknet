#include "detectorCommon.h"

using std::string;


struct FetcherThreadArgs
{
    image inputImage, inputImageScaled;
    string filename;
    CvCapture* capture;
    network* net;
};

image get_image_from_stream(CvCapture *cap);
image ipl_to_image(IplImage* src);

void *fetch_in_thread(void *ptr)
{
    FetcherThreadArgs* args = (FetcherThreadArgs*)ptr;
    if (args->capture) {
        args->inputImage = get_image_from_stream(args->capture);
    } else {
        args->inputImage = load_image_color(const_cast<char*>(args->filename.c_str()),0,0);
    }
    args->inputImageScaled = resize_image(args->inputImage, args->net->w, args->net->h);
    return 0;
}

struct DetectThreadArgs
{
    // Changes for each detection
    int frameNumber;
    image inputImage, inputImageScaled;

    // Whether to write the file in output or not
    bool writeOutput;

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
    int modelFileIndex;
    int modelIndexInFile;
    string *outputFilename;
    SERIALIZATION_NAMESPACE::SequenceSerialization* outSeq;

};

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
        if (args->allowedClasses->empty()) {
            keepDetection = true;
        } else {
            for (std::size_t c = 0; c < args->allowedClasses->size(); ++c) {
                if ((*args->allowedClasses)[c] == label) {
                    keepDetection = true;
                    break;
                }
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

        roi = getHistogramWindowFromDetectionRect(roi);

        std::vector<double> histogramOut;

        RectI imageBounds;
        imageBounds.x1 = 0;
        imageBounds.y1 = 0;
        imageBounds.x2 = args->inputImage.w;
        imageBounds.y2 = args->inputImage.h;

        roi.intersect(imageBounds, &roi);
        computeHistograms(roi, args->inputImage, &histogramOut);

        const std::size_t dataSize = sizeof(double) * histogramOut.size();
        args->modelFile->write((const char*)&histogramOut[0], dataSize);

        detection.modelIndex = args->modelIndexInFile;
        ++args->modelIndexInFile;
        detection.fileIndex = args->modelFileIndex;
#endif

        frameSerialization.detections.push_back(detection);

    }

    free_image(args->inputImage);


    // No detection, do not write to the file
    if (frameSerialization.detections.empty()) {
        std::cerr << "No detection at frame " << args->frameNumber << std::endl;
    } else {
        args->outSeq->frames.insert(std::make_pair(args->frameNumber, frameSerialization));
    }


    // Write the results file
    if (args->writeOutput) {
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
    }

    return 0;
} // detect_in_thread


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

static bool isVideoFile(const std::string& ext)
{
    std::string lower;
    for (std::size_t i = 0; i < ext.size(); ++i) {
        lower.push_back(std::tolower(ext[i]));
    }
    return lower == "mp4" || lower == "mov" || lower == "avi";
}

static void parseArguments(const std::list<string>& args,
                            std::vector<std::pair<std::string,int> > * inputFilesInOrder,
                           std::string *firstFrameFileName,
                           int* firstFrame, int* lastFrame,
                           CvCapture** videoStream,
                           int* numFrames,
                           string* cfgFilename,
                           string* weightFilename,
                           float* thresh,
                           std::vector<string>* names,
                           string* outputFilename)
{
    std::list<string> localArgs = args;

    SequenceParsing::SequenceFromPattern inputSequence;
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "-i");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("An input file must be specified with the -i switch.");
        }
        foundInput = localArgs.erase(foundInput);
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("-i switch without a file name");
        } else {
            SequenceParsing::filesListFromPattern_slow(*foundInput, &inputSequence);
            localArgs.erase(foundInput);
            if (inputSequence.empty()) {
                 throw std::invalid_argument("Empty input file");
            }
        }
    }

    bool isVideo;
    {
        *firstFrameFileName = inputSequence.begin()->second.begin()->second;
        std::string ext;
        std::size_t foundDot = firstFrameFileName->find_last_of(".");
        if (foundDot != std::string::npos) {
            ext = firstFrameFileName->substr(foundDot + 1);
        }
        isVideo = isVideoFile(ext);
    }


    int defaultFirstFrame,defaultLastFrame;
    if (!isVideo) {
        defaultFirstFrame = inputSequence.begin()->first;
        defaultLastFrame = inputSequence.rbegin()->first;
        *numFrames = defaultLastFrame - defaultFirstFrame + 1;

    } else {
        *videoStream = cvCaptureFromFile(firstFrameFileName->c_str());
        if (!videoStream) {
            throw std::invalid_argument("Could not open " + *firstFrameFileName);
        }

        *numFrames = cvGetCaptureProperty(*videoStream, CV_CAP_PROP_FRAME_COUNT);

        defaultFirstFrame = 1;
        defaultLastFrame = *numFrames - 1;

    }
    {
        std::list<string>::iterator foundInput = hasToken(localArgs, "--range");
        if (foundInput == localArgs.end()) {
            *firstFrame = defaultFirstFrame;
            *lastFrame = defaultLastFrame;
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


                if (*firstFrame < defaultFirstFrame || *lastFrame > defaultLastFrame) {
                    printf("Frame range must be in the sequence range (%i-%i)",defaultFirstFrame, defaultLastFrame);
                    exit(1);
                }
                localArgs.erase(foundInput);
            }
        }
        
    }
    {
        SequenceParsing::SequenceFromPattern::iterator startIt = inputSequence.find(*firstFrame);
        SequenceParsing::SequenceFromPattern::iterator endIt = inputSequence.find(*lastFrame);
        if (startIt == inputSequence.end() || endIt == inputSequence.end()) {
            throw std::invalid_argument("Invalid frame range");
        }
        ++endIt;
        for (SequenceParsing::SequenceFromPattern::iterator it = startIt; it != endIt; ++it) {
            inputFilesInOrder->push_back(std::make_pair(it->second.begin()->second, it->first));
        }
        *numFrames = inputFilesInOrder->size();

        if (inputFilesInOrder->empty()) {
            throw std::invalid_argument("At least one frame must be detected");
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

struct CaptureHolder
{
    CvCapture* _cap;
    CaptureHolder(CvCapture* cap)
    : _cap(cap)
    {

    }

    ~CaptureHolder()
    {
        if (_cap) {
            cvReleaseCapture(&_cap);
        }
    }
};


int renderImageSequence_main(int argc, char** argv)
{
    std::list<string> arguments;
    for (int i = 0; i < argc; ++i) {
        arguments.push_back(string(argv[i]));
    }

    CvCapture * videoStream = 0;
    std::vector<std::pair<std::string,int> > inputFilesInOrder;
    std::string firstFrameFileName;
    int firstFrame, lastFrame;
    int numFrames;
    string cfgFilename, weightFilename;
    float thresh;
    std::vector<string> names;
    string outputFile;
    float hier_thresh = .5;
    std::vector<string> allowedNames;
    allowedNames.push_back("person");
    parseArguments(arguments, &inputFilesInOrder, &firstFrameFileName, &firstFrame, &lastFrame, &videoStream, &numFrames, &cfgFilename, &weightFilename, &thresh, &names, &outputFile);

    CaptureHolder captureHolder(videoStream);


    // vector of pair<filename, framenumber> >


    SERIALIZATION_NAMESPACE::SequenceSerialization detectionResults;

    detectionResults.histogramSizes.push_back(HUE_HISTOGRAM_NUM_BINS);
    detectionResults.histogramSizes.push_back(SAT_HISTOGRAM_NUM_BINS);


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
    std::string modelAbsoluteFileName;
    {
        // Remove extension
        std::size_t foundLastDot = modelFileName.find_last_of(".");
        if (foundLastDot != std::string::npos) {
            modelFileName = modelFileName.substr(0, foundLastDot);
        }
        modelFileName += "_model";
        modelAbsoluteFileName = modelFileName;

        // Remove path
        std::size_t foundSlash = modelFileName.find_last_of("/");
        if (foundLastDot != std::string::npos) {
            modelFileName = modelFileName.substr(foundSlash + 1);
        }
    }



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
    detectArgs.modelFileIndex = 0;
    detectArgs.outSeq = &detectionResults;
    detectArgs.outputFilename = &outputFile;
    detectArgs.allowedClasses = &allowedNames;

    std::auto_ptr<FStreamsSupport::ofstream> modelFile;

    int curFrame_i = 0;

    while (curFrame_i < numFrames) {

        int frameNumber = !videoStream ? inputFilesInOrder[curFrame_i].second : curFrame_i + 1;
        const std::string& filename = !videoStream ? inputFilesInOrder[curFrame_i].first : firstFrameFileName;
        fetchArgs.capture = videoStream;
        fetchArgs.filename = filename;
        fetch_in_thread(&fetchArgs);

        if (!fetchArgs.inputImage.data) {
            throw std::invalid_argument("Could not open " + filename);
        }

        if (videoStream) {
            // Respect user frame range
            if (curFrame_i < firstFrame - 1) {
                continue;
            }
            if (curFrame_i > lastFrame - 1) {
                break;
            }
        }

        if (!modelFile.get() || detectArgs.modelIndexInFile >= MAX_NUM_MODELS_PER_FILE) {
            if (modelFile.get()) {
                modelFile->flush();
                modelFile->close();
                ++detectArgs.modelFileIndex;
            }
            detectArgs.modelIndexInFile = 0;
            modelFile.reset(new FStreamsSupport::ofstream);

            std::string actualFileName;
            {
                std::stringstream ss;
                ss << modelAbsoluteFileName << detectArgs.modelFileIndex +1;
                actualFileName = ss.str();
            }
            std::string actualRelativeFileName;
            {
                std::stringstream ss;
                ss << modelFileName << detectArgs.modelFileIndex +1;
                actualRelativeFileName = ss.str();
            }
            FStreamsSupport::open(modelFile.get(), actualFileName, std::ios_base::out | std::ios_base::trunc);
            if (!modelFile->is_open()) {
                throw std::invalid_argument("Could not open " + actualFileName);
            }


            detectionResults.modelFiles.push_back(actualRelativeFileName);
            detectArgs.modelFile = modelFile.get();
        }

        // Write if we reach the last one
        detectArgs.writeOutput = (curFrame_i == numFrames - 1 || curFrame_i == lastFrame - 1);
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
    if (modelFile.get()) {
        modelFile->flush();
        modelFile->close();
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
