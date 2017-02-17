/* ***** BEGIN LICENSE BLOCK *****
 * Copyright (C) 2017 INRIA
 * Author: Alexandre Gauthier-Foichat
 * ***** END LICENSE BLOCK ***** */

#ifndef SERIALIZATIONFWD_H
#define SERIALIZATIONFWD_H


#include <list>
#include <string>


#define SERIALIZATION_NAMESPACE Serialization
// Macros to use in each file to enter and exit the right name spaces.
#define SERIALIZATION_NAMESPACE_ENTER namespace SERIALIZATION_NAMESPACE {
#define SERIALIZATION_NAMESPACE_EXIT }
#define SERIALIZATION_NAMESPACE_USING using namespace SERIALIZATION_NAMESPACE;

SERIALIZATION_NAMESPACE_ENTER;


SERIALIZATION_NAMESPACE_EXIT;

#ifndef YAML
#error "YAML should be defined to YAML_AUTOCAM"
#endif

namespace YAML {
    class Emitter;
    class Node;
}

#endif // SERIALIZATIONFWD_H
