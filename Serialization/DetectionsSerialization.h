/* ***** BEGIN LICENSE BLOCK *****
 * Copyright (C) 2017 INRIA
 * Author: Alexandre Gauthier-Foichat
 * ***** END LICENSE BLOCK ***** */

#ifndef DETECTIONS_SERIALIZATION_H
#define DETECTIONS_SERIALIZATION_H

#include <string>
#include <list>
#include <vector>
#include <map>
#include "SerializationBase.h"


SERIALIZATION_NAMESPACE_ENTER;

#define SERIALIZATION_FILE_FORMAT_HEADER "# Detections"

class DetectionSerialization : public SerializationObjectBase
{
public:

    // Detection rectangle in non-normalized coordinates
    // (x1,y1) being coordinates of the bottom left corner
    // (x2,y2) being coordinates of the top right corner
    double x1, y1, x2, y2;

    // Detection score, normalized in 0 <= score <= 1
    double score;

    // An identifier of the object type, e.g: "dog", "person", "billy" ...
    std::string label;

    // An optional label used in the user interface to label the detection to replace
    // the identifier. If unset, this will be assumed to be the identifier.
    std::string uiLabel;

    // The file offset at which the model data for this detection starts
    std::size_t dataFileOffset;

public:

    DetectionSerialization()
    : x1(0)
    , y1(0)
    , x2(0)
    , y2(0)
    , score(0)
    , label()
    , uiLabel()
    , dataFileOffset(0)
    {

    }

    virtual ~DetectionSerialization()
    {

    }


    virtual void encode(YAML::Emitter& em) const;

    virtual void decode(const YAML::Node& node);
};

class FrameSerialization : public SerializationObjectBase
{
public:

    std::list<DetectionSerialization> detections;

    FrameSerialization()
    : detections()
    {

    }

    virtual ~FrameSerialization()
    {

    }

    virtual void encode(YAML::Emitter& em) const;

    virtual void decode(const YAML::Node& node);
};

class SequenceSerialization : public SerializationObjectBase
{
public:

    std::map<int, FrameSerialization> frames;

    // Hue and saturation histogram of the image for the detection rectangle
    std::vector<int> histogramSizes;

    // The filename of the model file: this is relative to the actual detection fle
    std::string histogramFileName;


    SequenceSerialization()
    : frames()
    , histogramSizes()
    , histogramFileName()
    {

    }

    virtual ~SequenceSerialization()
    {

    }

    virtual void encode(YAML::Emitter& em) const;

    virtual void decode(const YAML::Node& node);
};

SERIALIZATION_NAMESPACE_EXIT;

#endif // DETECTIONS_SERIALIZATION_H
