/* ***** BEGIN LICENSE BLOCK *****
 * Copyright (C) 2017 INRIA
 * Author: Alexandre Gauthier-Foichat
 * ***** END LICENSE BLOCK ***** */



#include <string>
#include <sstream>
#include <iostream>
#include <list>


#ifdef _WIN32
#include <windows.h>
#endif

#include "FStreamsSupport.h"
#include "SerializationIO.h"
#include "DetectionsSerialization.h"

using namespace SERIALIZATION_NAMESPACE;
using namespace std;

typedef list<string> StringList;

static void
printUsage(const string& programName)
{

    /* Text must hold in 80 columns ************************************************/
    stringstream ss;
    ss << programName;
    ss << " usage:\n"
    "This program can convert ground-truth detections to the odg file format.\n"
    "Each line in the ground-truth file is assumed to be a rectangle x1,y1,x2,y2\n"
    "as defined in the README.md file in this directory representing the result\n"
    "of a detection for the object given in argument. \n"
    "\n\n"
    "Program options:\n\n"
    "-i <label> [<UILabel>] <filename>: Add a new detection file, prepended with the\n"
    "                            label of the detection and optionaly a UI label.\n"
    "                            This option may be specified multiple times.\n\n"
    "-o <filename>: The output .odg file to write. Any existing file will be\n"
    "               overwritten.\n\n"
    "--start <number>: Specifies the first frame in the file. Default is 1\n\n"
    "--step <number>: Specified how many frames to skip between frames. Default is 1\n\n";
    cout << ss.str() << endl;
} // printUsage

static StringList::iterator hasToken(StringList &localArgs, const string& token)
{
    for (StringList::iterator it = localArgs.begin(); it!=localArgs.end(); ++it) {
        if (*it == token) {
            return it;
        }
    }
    return localArgs.end();
} // hasToken

struct InputFile
{
    string filename, identifier, label;
};

static void parseArgs(const StringList& appArgs, list<InputFile>* inputFiles, string* outputFile, int* start, int* step)
{
    StringList localArgs = appArgs;

    *start = 1;
    *step = 1;

    // Find input files
    {

        StringList::iterator foundInput = hasToken(localArgs, "-i");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("At least one input file must be specified with the -i switch.");
        }
        while (foundInput != localArgs.end()) {
            if (foundInput == localArgs.end()) {
                if (inputFiles->empty()) {
                    throw std::invalid_argument("At least one input file must be specified with the -i switch.");
                } else {
                    break;
                }
            }

            InputFile input;
            foundInput = localArgs.erase(foundInput);
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("-i switch without an identifier");
            }

            input.identifier = *foundInput;
            foundInput = localArgs.erase(foundInput);

            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("-i switch without a file");
            }

            // Check if this is a label or the file
            StringList::iterator hasNext = foundInput;
            ++hasNext;
            if (hasNext == localArgs.end() || (!hasNext->empty() && hasNext->substr(0, 1) == "-")) {
                // No label, just a file
                input.filename = *foundInput;
                foundInput = localArgs.erase(foundInput);
            } else {
                input.label = *foundInput;
                foundInput = localArgs.erase(foundInput);
                input.filename = *foundInput;
                foundInput = localArgs.erase(foundInput);
            }
            inputFiles->push_back(input);

            foundInput = hasToken(localArgs, "-i");
        }
    }
    {
        StringList::iterator foundInput = hasToken(localArgs, "-o");
        if (foundInput == localArgs.end()) {
            throw std::invalid_argument("An output file must be specified with the -o switch.");
        }
        ++foundInput;
        if ( foundInput == localArgs.end() ) {
            throw std::invalid_argument("-o switch without a file name");
        } else {
            *outputFile = *foundInput;
            localArgs.erase(foundInput);
        }
    }
    {
        StringList::iterator foundInput = hasToken(localArgs, "--start");
        if (foundInput != localArgs.end()) {
            ++foundInput;
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--start switch without a frame number");
            } else {
                std::istringstream ss;
                ss.str(*foundInput);
                ss >> *start;
                localArgs.erase(foundInput);
            }
        }
    }
    {
        StringList::iterator foundInput = hasToken(localArgs, "--step");
        if (foundInput != localArgs.end()) {
            ++foundInput;
            if ( foundInput == localArgs.end() ) {
                throw std::invalid_argument("--step switch without a frame number");
            } else {
                std::istringstream ss;
                ss.str(*foundInput);
                ss >> *step;

                if (*step == 0) {
                    throw std::invalid_argument("The --step argument cannot be 0");
                }
                localArgs.erase(foundInput);
            }
        }
    }
} // parseArgs



static void parseInputFile(const InputFile& file, int startFrame, int frameStep, SequenceSerialization* sequence)
{
    FStreamsSupport::ifstream istream;
    FStreamsSupport::open(&istream, file.filename);
    if (!istream) {
        std::stringstream ss;
        ss << "Failed to open file:" << file.filename;
        throw std::runtime_error(ss.str());
    }

    int f = startFrame;
    std::string buf;
    while (istream.good()) {
        std::getline(istream, buf);

        // Allow empty lines
        if (buf.empty()) {
            continue;
        }

        DetectionSerialization detection;
        detection.label = file.identifier;
        detection.uiLabel = file.label;
        detection.score = 1.;

        std::istringstream ss(buf);
        if (!(ss >> detection.x1)) {
            throw std::runtime_error("Invalid file: " + file.filename);
        }
        if (!(ss >> detection.y1)) {
            throw std::runtime_error("Invalid file: " + file.filename);
        }
        if (!(ss >> detection.x2)) {
            throw std::runtime_error("Invalid file: " + file.filename);
        }
        if (!(ss >> detection.y2)) {
            throw std::runtime_error("Invalid file: " + file.filename);
        }

        FrameSerialization& frameData = sequence->frames[f];
        frameData.detections.push_back(detection);
        f += frameStep;
    }
} // parseInputFile



#ifdef _WIN32
// g++ knows nothing about wmain
// https://sourceforge.net/p/mingw-w64/wiki2/Unicode%20apps/
// If it fails to compile it means either UNICODE or _UNICODE is not defined (it should be in global.pri) and
// the project is not linking against -municode
extern "C" {
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char *argv[])
#endif
{
    //QCoreApplication app(argc,argv);
    StringList arguments;
    for (int i = 0; i < argc; ++i) {
#ifdef _WIN32
        std::wstring ws(argv[i]);
        arguments.push_back(ws);
#else
        arguments.push_back(std::string(argv[i]));
#endif
    }
    assert(!arguments.empty());
    if (arguments.empty()) {
        return 1;
    }

    // Parse app args
    int frameStep, startFrame;
    list<InputFile> inputFiles;
    string outputFile;
    try {
        parseArgs(arguments, &inputFiles, &outputFile, &startFrame, &frameStep);
    } catch (const std::exception &e) {
        std::cerr << "Error while parsing command line arguments: " << e.what() << std::endl;
        printUsage(*arguments.begin());
        return 1;
    }
    assert(!inputFiles.empty());

    SequenceSerialization sequence;

    for (list<InputFile>::iterator it = inputFiles.begin(); it != inputFiles.end(); ++it) {
        try {
            parseInputFile(*it, startFrame, frameStep, &sequence);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return 1;
        }
    }

    FStreamsSupport::ofstream ofile;
    FStreamsSupport::open(&ofile, outputFile);
    if (!ofile) {
        std::cerr << "Could not open " << outputFile << std::endl;
        return 1;
    }

    try {
        write(ofile, sequence, SERIALIZATION_FILE_FORMAT_HEADER);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} // main
#ifdef _WIN32
} // extern "C"
#endif
