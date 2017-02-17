/* ***** BEGIN LICENSE BLOCK *****
 * Copyright (C) 2017 INRIA
 * Author: Alexandre Gauthier-Foichat
 * ***** END LICENSE BLOCK ***** */

#ifndef SERIALIZATIONIO_H
#define SERIALIZATIONIO_H


#include <istream>
#include <ostream>
#include <stdexcept>
#include <locale>

#include <yaml-cpp/emitter.h>
#include <yaml-cpp/node/impl.h>
#include <yaml-cpp/node/parse.h>

#include "SerializationFwd.h"

SERIALIZATION_NAMESPACE_ENTER
/**
 * @brief Write any serialization object to a YAML encoded file.
 * @param header The given header string will be written on the first line of the file unless it is empty.
 **/
template <typename T>
void write(std::ostream& stream, const T& obj, const std::string& header)
{
    if (!header.empty()) {
        stream << header.c_str() << "\n";
    }
    YAML::Emitter em;
    obj.encode(em);
    stream << em.c_str();
}

class InvalidSerializationFileException : public std::exception
{
    std::string _what;

public:

    InvalidSerializationFileException()
    : _what()
    {
    }

    virtual ~InvalidSerializationFileException() throw()
    {
    }

    virtual const char * what () const throw ()
    {
        return _what.c_str();
    }
};

/**
 * @brief Read any serialization object from a YAML encoded file. Upon failure an exception is thrown.
 * @param header The first line of the file is matched against the given header string.
 * If it does not match, this function throws a InvalidSerializationFileException exception
 * If header is empty, it does not check against the header.
 **/
template <typename T>
void read(const std::string& header, std::istream& stream, T* obj)
{
    if (!obj) {
        throw std::invalid_argument("Invalid serialization object");
    }
    {
        std::string firstLine;
        std::getline(stream, firstLine);
        if (!header.empty()) {
            if (firstLine != header) {
                throw InvalidSerializationFileException();
            }
        } else {
            // Check if the first-line contains a #, because it may contain a header which we should skip
            bool skipFirstLine = true;
            for (std::size_t i = 0; i < firstLine.size(); ++i) {
                if (std::isspace(firstLine[i])) {
                    continue;
                }
                if (firstLine[i] != '#') {
                    skipFirstLine = false;
                    break;
                } else {
                    break;
                }
            }
            // Since we called getline, we must reset the stream
            if (!skipFirstLine) {
                stream.seekg(0);
            }
        }
    }
    YAML::Node node = YAML::Load(stream);
    obj->decode(node);
}

SERIALIZATION_NAMESPACE_EXIT

#endif // SERIALIZATIONIO_H
