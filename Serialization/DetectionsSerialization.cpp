/* ***** BEGIN LICENSE BLOCK *****
 * Copyright (C) 2017 INRIA
 * Author: Alexandre Gauthier-Foichat
 * ***** END LICENSE BLOCK ***** */

#include "DetectionsSerialization.h"

#include <cstring>
#include <yaml-cpp/yaml.h>

SERIALIZATION_NAMESPACE_ENTER;

void
DetectionSerialization::encode(YAML::Emitter& em) const
{
    em << YAML::Flow << YAML::BeginMap;
    em << YAML::Key << "Rect" << YAML::Value << YAML::Flow << YAML::BeginSeq << x1 << y1 << x2 << y2 << YAML::EndSeq;
    em << YAML::Key << "Score" << YAML::Value << score;
    em << YAML::Key << "Label" << YAML::Value << label;
    if (!uiLabel.empty()) {
        em << YAML::Key << "UILabel" << YAML::Value << uiLabel;
    }

    if (fileIndex != -1) {
        em << YAML::Key << "ModelFile" << YAML::Value << fileIndex;
        em << YAML::Key << "ModelIndex" << YAML::Value << modelIndex;
    }
    em << YAML::EndMap;
}

void
DetectionSerialization::decode(const YAML::Node& node)
{
    if (!node.IsMap()) {
        throw YAML::InvalidNode();
    }
    YAML::Node rectNode = node["Rect"];
    if (!rectNode.IsSequence() || rectNode.size() != 4) {
        throw YAML::InvalidNode();
    }
    x1 = rectNode[0].as<double>();
    y1 = rectNode[1].as<double>();
    x2 = rectNode[2].as<double>();
    y2 = rectNode[3].as<double>();

    score = node["Score"].as<double>();

    label = node["Label"].as<std::string>();
    if (node["UILabel"]) {
        uiLabel = node["UILabel"].as<std::string>();
    }
    if (node["ModelFile"]) {
        fileIndex = node["ModelFile"].as<int>();
        modelIndex = node["ModelIndex"].as<std::size_t>();
    }

}

void
FrameSerialization::encode(YAML::Emitter& em) const
{
    if (detections.empty()) {
        return;
    }
    em << YAML::BeginSeq;
    for (std::list<DetectionSerialization>::const_iterator it = detections.begin(); it != detections.end(); ++it) {
        it->encode(em);
    }
    em << YAML::EndSeq;
}

void
FrameSerialization::decode(const YAML::Node& node)
{
    if (!node.IsSequence()) {
        throw YAML::InvalidNode();
    }
    for (std::size_t i = 0; i < node.size(); ++i) {
        DetectionSerialization s;
        s.decode(node[i]);
        detections.push_back(s);
    }
}

void
SequenceSerialization::encode(YAML::Emitter& em) const
{
    if (frames.empty()) {
        return;
    }

    em << YAML::BeginMap;


    if (!histogramSizes.empty()) {
        em << YAML::Key << "HistSizes" << YAML::Value << YAML::BeginSeq;
        for (std::size_t i = 0; i < histogramSizes.size(); ++i) {
            em << histogramSizes[i];
        }

        em << YAML::EndSeq;
    }

    if (!modelFiles.empty()) {
        em << YAML::Key << "ModelFiles" << YAML::Value << YAML::BeginSeq;
        for (std::size_t i = 0; i < modelFiles.size(); ++i) {
            em << modelFiles[i];
        }

        em << YAML::EndSeq;
    }

    em << YAML::Key << "Frames" << YAML::Value;
    em << YAML::BeginMap;
    for (std::map<int, FrameSerialization>::const_iterator it = frames.begin(); it != frames.end(); ++it) {
        em << YAML::Key << it->first;
        em << YAML::Value;
        it->second.encode(em);
    }

    em << YAML::EndMap;
    em << YAML::EndMap;

}

void
SequenceSerialization::decode(const YAML::Node& node)
{
    if (!node.IsMap()) {
        throw YAML::InvalidNode();
    }
    if (node["HistSizes"]) {
        YAML::Node histSizeNode = node["HistSizes"];
        histogramSizes.resize(histSizeNode.size());
        for (std::size_t i = 0; i < histSizeNode.size(); ++i) {
            histogramSizes[i] = histSizeNode[i].as<int>();
        }
    }

    if (node["ModelFiles"]) {
        YAML::Node filesNode = node["ModelFiles"];
        modelFiles.resize(filesNode.size());
        for (std::size_t i = 0; i < filesNode.size(); ++i) {
            modelFiles[i] = filesNode[i].as<std::string>();
        }
    }

    YAML::Node framesNode = node["Frames"];
    if (framesNode) {
        for (YAML::const_iterator it = framesNode.begin(); it != framesNode.end(); ++it) {
            int frame = it->first.as<int>();
            FrameSerialization& s = frames[frame];
            s.decode(it->second);
        }
    }

}

SERIALIZATION_NAMESPACE_EXIT;
